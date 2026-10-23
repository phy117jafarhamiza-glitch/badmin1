"""استخراج نقاط الجسم من الفيديو بواسطة MediaPipe.

يدعم واجهتين:
  1) legacy:  mediapipe <= 0.10.14 (mp.solutions.pose) والنموذج مضمَّن داخل الحزمة. هذا هو الخيار
     الموصى به للنشر (مثبَّت في requirements.txt).
  2) tasks:   mediapipe الأحدث، ويحتاج ملف النموذج pose_landmarker_lite.task في مجلد models/
     (يُنزَّل تلقائياً عند أول استخدام إن توفر الإنترنت).

تُحفظ المؤشرات فقط، ولا يُحفظ الفيديو.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .config import MAX_VIDEO_SECONDS, MODEL_DIR
from .metrics import SKELETON

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
MODEL_PATH = MODEL_DIR / "pose_landmarker_lite.task"


class PoseUnavailable(RuntimeError):
    pass


@dataclass
class PoseResult:
    kp: np.ndarray          # (T,33,4)  x,y بالبكسل (للإطار بعد التصغير)، z، visibility
    fps: float
    width: int
    height: int
    n_frames: int
    detected_ratio: float


def backend_status() -> tuple[str, str]:
    """(اسم الواجهة | none , رسالة)."""
    try:
        import mediapipe as mp
    except Exception as e:  # noqa: BLE001
        return "none", f"مكتبة mediapipe غير مثبتة ({e})"
    if getattr(mp, "solutions", None) is not None:
        return "legacy", f"mediapipe {mp.__version__} (النموذج مضمّن)"
    if MODEL_PATH.exists():
        return "tasks", f"mediapipe {mp.__version__} مع نموذج محلي"
    return "tasks-nomodel", f"mediapipe {mp.__version__}: ملف النموذج غير موجود وسيُنزَّل عند الحاجة"


def ensure_model(path: Path = MODEL_PATH) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
    except Exception as e:  # noqa: BLE001
        if path.exists():
            path.unlink()
        raise PoseUnavailable(
            "تعذر تنزيل نموذج تقدير الوضعية. نزّل الملف pose_landmarker_lite.task يدوياً وضعه في "
            f"المجلد models/ ({e})") from e
    return path


def _open_landmarker(backend: str, complexity: int = 1):
    """يعيد (دالة تحليل الإطار، دالة الإغلاق)."""
    import mediapipe as mp
    if backend == "legacy":
        pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=complexity,
                                      smooth_landmarks=True, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        def run(rgb, ts_ms):
            res = pose.process(rgb)
            if not res.pose_landmarks:
                return None
            return [(l.x, l.y, l.z, l.visibility) for l in res.pose_landmarks.landmark]
        return run, pose.close

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model = ensure_model()
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    lm = vision.PoseLandmarker.create_from_options(opts)
    last = {"t": -1}

    def run(rgb, ts_ms):
        ts = max(int(ts_ms), last["t"] + 1)
        last["t"] = ts
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = lm.detect_for_video(img, ts)
        if not res.pose_landmarks:
            return None
        return [(l.x, l.y, l.z, getattr(l, "visibility", 1.0) or 0.0) for l in res.pose_landmarks[0]]
    return run, lm.close


def video_info(path: str) -> dict:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise PoseUnavailable("تعذر فتح ملف الفيديو")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return dict(fps=fps, frames=n, width=w, height=h, seconds=(n / fps if fps else 0.0))


def prep_frame(frame, crop: Optional[tuple] = None, target_width: int = 640):
    """القص ثم التصغير: نفس المعالجة المستخدمة قبل تقدير الوضعية (لتطابق الإحداثيات عند الرسم)."""
    import cv2
    if crop:
        h0, w0 = frame.shape[:2]
        x0, y0, x1, y1 = crop
        frame = frame[int(y0 * h0):max(int(y1 * h0), int(y0 * h0) + 1), int(x0 * w0):max(int(x1 * w0), int(x0 * w0) + 1)]
    h0, w0 = frame.shape[:2]
    if w0 > target_width:
        scale = target_width / float(w0)
        frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
    return frame


def extract_keypoints(path: str, fps_override: Optional[float] = None, target_width: int = 640,
                      crop: Optional[tuple] = None, max_seconds: float = MAX_VIDEO_SECONDS,
                      progress: Optional[Callable[[float], None]] = None) -> PoseResult:
    """crop = (x0, y0, x1, y1) كنسب من 0 إلى 1 لحصر التحليل في منطقة اللاعب فقط."""
    import cv2
    backend, msg = backend_status()
    if backend == "none":
        raise PoseUnavailable(msg)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise PoseUnavailable("تعذر فتح ملف الفيديو")
    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (5 <= fps <= 480):
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = int(max_seconds * fps)
    run, close = _open_landmarker("legacy" if backend == "legacy" else "tasks")
    frames_kp, found = [], 0
    W = H = 0
    try:
        i = 0
        while i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = prep_frame(frame, crop, target_width)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = run(rgb, i * 1000.0 / fps)
            arr = np.full((33, 4), np.nan)
            arr[:, 3] = 0.0
            if lms is not None:
                found += 1
                a = np.array(lms, dtype=float)
                a[:, 0] *= W
                a[:, 1] *= H
                arr = a
            frames_kp.append(arr)
            i += 1
            if progress and total:
                progress(min(1.0, i / min(total, max_frames)))
    finally:
        close()
        cap.release()
    if not frames_kp:
        raise PoseUnavailable("لم تُقرأ أي إطارات من الفيديو")
    kp = np.stack(frames_kp)
    return PoseResult(kp=kp, fps=fps, width=W, height=H, n_frames=len(kp),
                      detected_ratio=found / len(kp))


def read_roi_brightness(path: str, roi: tuple, fps_override: Optional[float] = None,
                        max_seconds: float = MAX_VIDEO_SECONDS) -> tuple[np.ndarray, float]:
    """متوسط السطوع داخل منطقة (نسب x0,y0,x1,y1) لكل إطار – لاكتشاف لحظة ظهور المثير."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise PoseUnavailable("تعذر فتح ملف الفيديو")
    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    vals = []
    while len(vals) < int(max_seconds * fps):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = roi
        patch = frame[int(y0 * h):max(int(y1 * h), int(y0 * h) + 1), int(x0 * w):max(int(x1 * w), int(x0 * w) + 1)]
        vals.append(float(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).mean()))
    cap.release()
    return np.array(vals), fps


def get_frame(path: str, index: int):
    import cv2
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def draw_skeleton(frame_bgr, xy_frame: np.ndarray, scale: float = 1.0, labels: Optional[dict] = None):
    """يرسم الهيكل على إطار؛ labels = {landmark_index: 'نص لاتيني/رقمي'} (cv2 لا يدعم العربية)."""
    import cv2
    img = frame_bgr.copy()
    for a, b in SKELETON:
        pa, pb = xy_frame[a] * scale, xy_frame[b] * scale
        cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), (0, 220, 0), 3, cv2.LINE_AA)
    for j in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(img, tuple(int(v) for v in xy_frame[j] * scale), 5, (0, 120, 255), -1, cv2.LINE_AA)
    if labels:
        for j, txt in labels.items():
            p = xy_frame[j] * scale
            cv2.putText(img, txt, (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (int(p[0]) + 8, int(p[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)
    return img


def open_stream():
    """محلّل إطارات للبث المباشر: run(bgr) -> مصفوفة (33,2) بالبكسل أو None، و close()."""
    import cv2
    backend, msg = backend_status()
    if backend == "none":
        raise PoseUnavailable(msg)
    run, close = _open_landmarker("legacy" if backend == "legacy" else "tasks", complexity=0)
    t0 = {"n": 0}

    def analyze(bgr):
        h, w = bgr.shape[:2]
        t0["n"] += 1
        lms = run(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t0["n"] * 33.0)
        if lms is None:
            return None
        a = np.array(lms, dtype=float)
        if np.nanmean(a[[11, 12, 23, 24], 3]) < 0.4:      # الجذع غير ظاهر بوضوح
            return None
        a[:, 0] *= w
        a[:, 1] *= h
        return a[:, :2]
    return analyze, close
