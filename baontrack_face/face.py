"""
face.py — face detection and verification for BaonTrack.

Two small ONNX models that ship through OpenCV (no dlib, no TensorFlow,
no GPU — so it runs on Colab's CPU and on Streamlit Community Cloud):

  * YuNet  (FaceDetectorYN)   finds faces + 5 landmarks in an image
  * SFace  (FaceRecognizerSF) turns an aligned face into a 128-number
                              "embedding" (a face fingerprint)

Verification = compare the embedding of a fresh photo against the
embeddings stored at enrollment, using cosine similarity. We store ONLY
the embeddings (never the photos), and an embedding can't be turned back
into a picture of the face.

Needs opencv-python-headless >= 4.8.
"""

import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"

_MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"

# Cosine-similarity cutoff recommended by the SFace authors: >= 0.363 means
# "same person". Raise it (e.g. 0.45) to be stricter, lower it to be looser.
MATCH_THRESHOLD = 0.363

MIN_DETECTION_SCORE = 0.85   # detector confidence needed to accept a face for enroll/verify
MIN_FACE_WIDTH_PX = 80       # too-small faces give unreliable embeddings

# cv2 network objects are not thread-safe, and the live-video callback runs on
# a different thread than the Streamlit script, so every model call takes this lock.
_lock = threading.Lock()
_detector = None
_recognizer = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def ensure_models() -> None:
    """Download the two ONNX files on first run (about 40 MB total)."""
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in _MODEL_URLS.items():
        path = MODELS_DIR / name
        if path.exists() and path.stat().st_size > 100_000:
            continue
        urllib.request.urlretrieve(url, path)
        if path.stat().st_size < 100_000:  # a Git-LFS pointer file, not the real model
            path.unlink()
            raise RuntimeError(f"Download of {name} failed (got a pointer file, not the model).")


def _get_detector():
    global _detector
    if _detector is None:
        ensure_models()
        _detector = cv2.FaceDetectorYN.create(
            str(MODELS_DIR / DETECTOR_FILE), "", (320, 320),
            score_threshold=0.8, nms_threshold=0.3, top_k=5000,
        )
    return _detector


def _get_recognizer():
    global _recognizer
    if _recognizer is None:
        ensure_models()
        _recognizer = cv2.FaceRecognizerSF.create(str(MODELS_DIR / RECOGNIZER_FILE), "")
    return _recognizer


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def decode_image(data: bytes) -> np.ndarray | None:
    """Bytes from st.camera_input / file upload -> BGR image."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def detect_faces(img_bgr: np.ndarray) -> np.ndarray:
    """Returns an (N, 15) array, one row per face:
    x, y, w, h, 5 landmark (x, y) pairs, detection score. Empty (0, 15) if none."""
    h, w = img_bgr.shape[:2]
    with _lock:
        det = _get_detector()
        det.setInputSize((w, h))
        _, faces = det.detect(img_bgr)
    return faces if faces is not None else np.zeros((0, 15), dtype=np.float32)


def draw_faces(img_bgr: np.ndarray, faces: np.ndarray, labels=None, colors=None) -> np.ndarray:
    """Draws a box + landmarks on each face (in place) and returns the image."""
    for i, f in enumerate(faces):
        x, y, w, h = f[:4].astype(int)
        color = colors[i] if colors else (0, 200, 0)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, 2)
        for lx, ly in f[4:14].reshape(5, 2).astype(int):
            cv2.circle(img_bgr, (lx, ly), 2, (0, 140, 255), -1)
        label = labels[i] if labels else f"{f[14]:.2f}"
        cv2.putText(img_bgr, label, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img_bgr


# ---------------------------------------------------------------------------
# Embeddings + matching
# ---------------------------------------------------------------------------

def embed_face(img_bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
    with _lock:
        rec = _get_recognizer()
        aligned = rec.alignCrop(img_bgr, face_row)
        feat = rec.feature(aligned)
    return np.asarray(feat, dtype=np.float32).reshape(1, -1)


def get_embedding(img_bgr: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Embedding for the single face in the image.
    Returns (embedding, None) on success or (None, human-readable problem)."""
    if img_bgr is None:
        return None, "Couldn't read that image."
    faces = detect_faces(img_bgr)
    if len(faces) == 0:
        return None, "No face found — face the camera in good light."
    if len(faces) > 1:
        return None, "More than one face in the frame — make sure only you are visible."
    face = faces[0]
    if face[14] < MIN_DETECTION_SCORE:
        return None, "Face isn't clear enough — try better lighting."
    if face[2] < MIN_FACE_WIDTH_PX:
        return None, "Face is too small — move closer to the camera."
    return embed_face(img_bgr, face), None


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_score(embedding: np.ndarray, stored: list[np.ndarray]) -> float:
    """Highest cosine similarity between `embedding` and any enrolled sample."""
    return max((cosine_score(embedding, s) for s in stored), default=-1.0)


def is_match(embedding: np.ndarray, stored: list[np.ndarray]) -> tuple[bool, float]:
    score = best_score(embedding, stored)
    return score >= MATCH_THRESHOLD, score


# Storing embeddings in SQLite: float32 array <-> bytes
def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=np.float32).reshape(-1).tobytes()


def embedding_from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
