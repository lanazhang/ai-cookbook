"""Semantic video search with Qwen3-VL-Embedding and FAISS: the whole pipeline in one script.

Needs: opencv-python-headless, faiss-cpu, numpy, requests, and an embedding server at BASE_URL that speaks the
OpenAI-style /v1/embeddings API (for example vLLM serving Qwen/Qwen3-VL-Embedding-8B).
"""
import base64, os, tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2, faiss, numpy as np, requests

BASE_URL, MODEL = "http://127.0.0.1:18001", "Qwen/Qwen3-VL-Embedding-8B"
VIDEO, SEGMENT_SEC, WIDTH, FPS, KEYFRAMES_PER_CLIP = "data/meridian.mp4", 5, 448, 2, 3

def embed(parts):
    """One multimodal input (a list of text / image / video parts) -> unit-length vector."""
    if all(p["type"] == "text" for p in parts):      # plain text queries use the simple `input` form
        body = {"model": MODEL, "encoding_format": "float", "input": " ".join(p["text"] for p in parts)}
    else:
        body = {"model": MODEL, "encoding_format": "float", "messages": [{"role": "user", "content": parts}]}
    r = requests.post(f"{BASE_URL}/v1/embeddings", json=body, timeout=300)
    r.raise_for_status()
    v = np.asarray(r.json()["data"][0]["embedding"], dtype="float32")
    return v / np.linalg.norm(v)                     # L2-normalise: inner product = cosine

b64 = lambda data: base64.b64encode(data).decode()
text_part = lambda s: {"type": "text", "text": s}

def image_part(frame):
    jpg = cv2.imencode(".jpg", frame)[1].tobytes()
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(jpg)}}

def video_part(frames):
    path = tempfile.mktemp(suffix=".mp4")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    data = open(path, "rb").read(); os.remove(path)
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + b64(data)}}

def small(frame):
    return cv2.resize(frame, (WIDTH, frame.shape[0] * WIDTH // frame.shape[1]))

def clip_frames(start, end):
    """Decode [start, end) seconds, keep FPS frames per second, downscale."""
    cap = cv2.VideoCapture(VIDEO)
    src = cap.get(cv2.CAP_PROP_FPS)
    first, step = int(start * src), max(int(src / FPS), 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for i in range(int(end * src) - first):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(small(f))
    cap.release()
    return frames

def frame_at(sec):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ok, f = cap.read()
    cap.release()
    return small(f)

cap = cv2.VideoCapture(VIDEO)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()

# fixed 5-second windows, each remembering its start and end time
segments = [{"row": n, "startSec": float(t), "endSec": float(min(t + SEGMENT_SEC, duration))}
            for n, t in enumerate(np.arange(0, duration - 1, SEGMENT_SEC))]
# a few still frames per clip, used for image queries
keyframes = [{"row": s["row"], "sec": s["startSec"] + (k + 0.5) / KEYFRAMES_PER_CLIP * (s["endSec"] - s["startSec"])}
             for s in segments for k in range(KEYFRAMES_PER_CLIP)]

with ThreadPoolExecutor(4) as pool:
    E = np.vstack(list(pool.map(lambda s: embed([video_part(clip_frames(s["startSec"], s["endSec"]))]), segments)))
    F = np.vstack(list(pool.map(lambda k: embed([image_part(frame_at(k["sec"]))]), keyframes)))

index = faiss.IndexFlatIP(E.shape[1]); index.add(E)               # clip vectors, row i = segments[i]
frame_index = faiss.IndexFlatIP(F.shape[1]); frame_index.add(F)   # keyframe vectors, row j = keyframes[j]

def search_clips(parts, k=5):
    """Text query -> nearest clip vectors."""
    q = embed(parts)
    scores, rows = index.search(q[None, :], k)
    return [{"score": float(s), **segments[r]} for s, r in zip(scores[0], rows[0])]

def search_frames(parts, k=5):
    """Image query -> nearest keyframe vectors; the best frame of each clip wins."""
    q = embed(parts)
    scores, ids = frame_index.search(q[None, :], k * KEYFRAMES_PER_CLIP * 4)
    best = {}
    for s, i in zip(scores[0], ids[0]):
        best.setdefault(keyframes[i]["row"], float(s))   # sorted, so the first hit is the best
    return [{"score": s, **segments[row]} for row, s in list(best.items())[:k]]


if __name__ == "__main__":
    for h in search_clips([text_part("two vintage police cars parked by the ocean")]):
        print(f"{h['startSec']:6.0f}s - {h['endSec']:6.0f}s  cosine {h['score']:.3f}")
