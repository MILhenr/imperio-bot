import os, uuid, base64, json, threading, time
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Futsal Tracker Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # o site (Railway) chama esse serviço de outro domínio
    allow_methods=["*"],
    allow_headers=["*"],
)

TMP_DIR = "/tmp/futsal_jobs"
os.makedirs(TMP_DIR, exist_ok=True)

# job_id -> {"status": "...", "result": {...} | None, "error": str | None, "video_path": str}
JOBS = {}

MAX_DURATION_SEC = 10.5  # até 10s de lance, com uma pequena margem

_yolo_model = None
def get_model():
    """Carrega o YOLO só na primeira vez que for usado (lazy load)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")  # baixa o modelo na primeira execução (precisa de internet)
    return _yolo_model


@app.get("/")
def health():
    return {"ok": True, "service": "futsal-tracker-backend"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Recebe o vídeo, valida duração, devolve o primeiro frame (base64) pro
    usuário marcar os 4 cantos da quadra."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(TMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, "input.mp4")

    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(400, "Não consegui abrir esse vídeo. Tenta um .mp4 ou .mov comum.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0

    if duration > MAX_DURATION_SEC:
        cap.release()
        raise HTTPException(400, f"Esse vídeo tem {duration:.1f}s. O limite é {MAX_DURATION_SEC:.0f}s — manda só o lance.")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "Não consegui ler o primeiro frame do vídeo.")

    h, w = frame.shape[:2]
    max_dim = 900
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode("ascii")

    JOBS[job_id] = {"status": "awaiting_corners", "result": None, "error": None,
                     "video_path": video_path, "duration": duration, "fps": fps,
                     "frame_scale": scale}

    return {"job_id": job_id, "first_frame_b64": b64, "duration": round(duration, 2)}


@app.post("/process/{job_id}")
def start_processing(job_id: str, corners: str = Form(...)):
    """Recebe os pontos de calibração marcados: cada um tem {x,y} (pixel no
    frame reduzido) e {fx,fy} (posição real na quadra, fração 0..1) — pode
    ser qualquer combinação de referências conhecidas da quadra, não só os 4
    cantos externos. Mínimo de 4 pontos."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado (ou expirou).")
    try:
        pts = json.loads(corners)
        assert len(pts) >= 4
        for p in pts:
            assert "x" in p and "y" in p and "fx" in p and "fy" in p
    except Exception:
        raise HTTPException(400, "Formato de pontos inválido — precisa de pelo menos 4 pontos {x,y,fx,fy}.")

    job["status"] = "processing"
    job["progress"] = 0
    t = threading.Thread(target=_run_pipeline, args=(job_id, pts), daemon=True)
    t.start()
    return {"status": "processing"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job não encontrado (ou expirou).")
    return {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _run_pipeline(job_id, corner_pts):
    job = JOBS[job_id]
    try:
        result = process_video(job["video_path"], corner_pts, job["frame_scale"],
                                progress_cb=lambda p: job.__setitem__("progress", p))
        job["result"] = result
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
# PIPELINE de verdade: extrai frames, detecta pessoas (YOLO), rastreia
# (ByteTrack via ultralytics), separa times por cor de uniforme, converte pra
# coordenada da quadra (homografia dos 4 cantos), gera o JSON no formato do app.
# ---------------------------------------------------------------------------
def process_video(video_path, calib_points, frame_scale, progress_cb=None):
    """calib_points: lista de {x, y, fx, fy} — x,y = pixel no frame reduzido;
    fx,fy = posição real na quadra (fração 0..1) que esse ponto representa.
    Não precisa ser os 4 cantos — pode ser qualquer combinação de pontos
    conhecidos (trave, marca de pênalti, centro do círculo, etc), desde que
    sejam pelo menos 4 e não estejam todos em linha reta."""
    model = get_model()

    src_pts = np.array([[p["x"], p["y"]] for p in calib_points], dtype=np.float32)
    dst_pts = np.array([[p["fx"], p["fy"]] for p in calib_points], dtype=np.float32)
    method = cv2.RANSAC if len(calib_points) > 4 else 0
    H, _ = cv2.findHomography(src_pts, dst_pts, method)
    if H is None:
        raise RuntimeError("Não consegui calcular a posição da quadra com esses pontos. Tenta marcar pontos mais espalhados (não todos numa linha reta).")

    # amostra a 6 fps pra não pesar demais (lance de até 10s -> até 60 frames)
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_every = max(1, round(src_fps / 6))
    cap.release()

    # ultralytics cuida do rastreamento (ByteTrack) frame a frame
    track_results = model.track(
        source=video_path, classes=[0], persist=True, tracker="bytetrack.yaml",
        vid_stride=sample_every, conf=0.25, verbose=False, stream=True
    )

    tracks = {}  # track_id -> lista de {frame_idx, cx, cy_foot}
    frame_idx = 0
    total_estimate = 60
    for r in track_results:
        frame_idx += 1
        if progress_cb: progress_cb(min(90, int(frame_idx / total_estimate * 70)))
        if r.boxes is None or r.boxes.id is None:
            continue
        frame_img = r.orig_img
        for box, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy()):
            x1, y1, x2, y2 = box
            tid = int(tid)
            foot_x, foot_y = (x1 + x2) / 2, y2  # base da caixa = pés
            jersey_color = sample_jersey_color(frame_img, x1, y1, x2, y2)
            tracks.setdefault(tid, []).append({
                "frame": frame_idx, "px": foot_x, "py": foot_y, "color": jersey_color
            })

    if progress_cb: progress_cb(92)

    # descarta rastros muito curtos (ruído)
    tracks = {tid: pts for tid, pts in tracks.items() if len(pts) >= 3}
    if not tracks:
        raise RuntimeError("Não detectei jogadores com confiança nesse vídeo.")

    # classifica cada rastro em 2 times, por cor média do uniforme (k-means, k=2)
    colors = np.array([np.mean([p["color"] for p in pts], axis=0) for pts in tracks.values()], dtype=np.float32)
    team_labels = kmeans2(colors, k=2)

    tids = list(tracks.keys())
    entities = []
    blue_num, red_num = 1, 1
    id_map = {}
    for i, tid in enumerate(tids):
        team = "blue" if team_labels[i] == 0 else "red"
        eid = i + 1
        id_map[tid] = eid
        if team == "blue":
            num = blue_num; blue_num += 1
        else:
            num = red_num; red_num += 1
        entities.append({"id": eid, "team": team, "number": num, "name": "", "heightM": 1.75})

    # amostra ~6-8 "etapas" ao longo do clipe (não frame a frame — fica mais
    # limpo pra edição manual depois)
    max_frame = max(p["frame"] for pts in tracks.values() for p in pts)
    num_steps = min(8, max(3, max_frame // 4))
    steps = []
    for s in range(num_steps):
        target_frame = round((s / (num_steps - 1)) * max_frame) if num_steps > 1 else 0
        positions = {}
        for tid, pts in tracks.items():
            nearest = min(pts, key=lambda p: abs(p["frame"] - target_frame))
            fx, fy = apply_homography(H, nearest["px"], nearest["py"])
            positions[str(id_map[tid])] = {
                "x": clamp01(fx), "y": clamp01(fy),
                "facingAngle": -1.5708, "customFacing": False
            }
        steps.append({"positions": positions})

    if progress_cb: progress_cb(100)

    return {
        "name": "Jogada importada do vídeo",
        "entities": entities,
        "steps": steps,
        "nextBlueNum": blue_num,
        "nextRedNum": red_num,
        "savedAt": int(time.time() * 1000),
    }


def sample_jersey_color(frame_img, x1, y1, x2, y2):
    """Pega a cor média da região do TORSO (metade de cima da caixa da
    pessoa), que costuma ser a camisa."""
    h, w = frame_img.shape[:2]
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0]
    torso_y2 = y1 + int((y2 - y1) * 0.55)
    region = frame_img[y1:max(y1 + 1, torso_y2), x1:x2]
    if region.size == 0:
        return [0, 0, 0]
    mean_bgr = region.reshape(-1, 3).mean(axis=0)
    return mean_bgr.tolist()


def kmeans2(data, k=2, iters=25):
    """K-means bem simples (evita depender de scikit-learn)."""
    if len(data) < k:
        return [0] * len(data)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(data), k, replace=False)
    centers = data[idx].copy()
    labels = np.zeros(len(data), dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            pts = data[labels == c]
            if len(pts) > 0:
                centers[c] = pts.mean(axis=0)
    return labels.tolist()


def apply_homography(H, px, py):
    pt = np.array([px, py, 1.0])
    mapped = H @ pt
    mapped /= mapped[2]
    return float(mapped[0]), float(mapped[1])


def clamp01(v):
    return max(0.0, min(1.0, v))
