"""حساب المؤشرات الحركية من نقاط الجسم (Keypoints).

كل الدوال هنا دوال رياضية خالصة (numpy) ولا تعتمد على mediapipe أو Streamlit،
لذلك يمكن اختبارها ببيانات اصطناعية.

شكل البيانات: kp  = مصفوفة (T, 33, 4) بالأعمدة [x, y, z, visibility] وإحداثيات x,y بالبكسل.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import RT_MAX_VALID, RT_MIN_VALID

LM = dict(nose=0, l_sh=11, r_sh=12, l_el=13, r_el=14, l_wr=15, r_wr=16,
          l_hip=23, r_hip=24, l_kn=25, r_kn=26, l_an=27, r_an=28)

SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
            (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]


# ----------------------------------------------------------------- تنظيف الإشارة
def _interp_nan_1d(a: np.ndarray) -> np.ndarray:
    a = a.astype(float).copy()
    bad = ~np.isfinite(a)
    if bad.all():
        return np.zeros_like(a)
    if bad.any():
        idx = np.arange(len(a))
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def _moving_average(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(a) < win:
        return a
    pad = win // 2
    ap = np.pad(a, ((pad, win - 1 - pad),) + ((0, 0),) * (a.ndim - 1), mode="edge")
    ker = np.ones(win) / win
    if a.ndim == 1:
        return np.convolve(ap, ker, mode="valid")
    out = np.empty_like(a, dtype=float)
    for j in range(a.shape[1]):
        out[:, j] = np.convolve(ap[:, j], ker, mode="valid")
    return out


def clean_keypoints(kp: np.ndarray, fps: float, vis_thr: float = 0.3) -> np.ndarray:
    """يعيد xy نظيفة (T,33,2): استبدال النقاط الضعيفة بالاستيفاء ثم تنعيم خفيف."""
    kp = np.asarray(kp, dtype=float)
    xy = kp[:, :, :2].copy()
    vis = kp[:, :, 3]
    xy[vis < vis_thr] = np.nan
    T = xy.shape[0]
    for j in range(xy.shape[1]):
        for c in range(2):
            xy[:, j, c] = _interp_nan_1d(xy[:, j, c])
    win = max(3, int(round(fps * 0.03)) | 1)
    for j in range(xy.shape[1]):
        xy[:, j, :] = _moving_average(xy[:, j, :], win)
    return xy


# -------------------------------------------------------------------- هندسة الجسم
def angle_deg(a, b, c) -> float:
    """الزاوية عند النقطة b بين a و c بالدرجات."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    ba, bc = a - b, c - b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(ba, bc) / den, -1, 1))))


def torso_length(xy: np.ndarray) -> float:
    mid_sh = (xy[:, LM["l_sh"]] + xy[:, LM["r_sh"]]) / 2
    mid_hip = (xy[:, LM["l_hip"]] + xy[:, LM["r_hip"]]) / 2
    d = np.linalg.norm(mid_sh - mid_hip, axis=1)
    t = float(np.median(d))
    return t if t > 1e-6 else 1.0


def _side(dominant: str) -> tuple[int, int, int]:
    return (LM["r_sh"], LM["r_el"], LM["r_wr"]) if dominant == "right" else (LM["l_sh"], LM["l_el"], LM["l_wr"])


def movement_signal(xy: np.ndarray, fps: float, torso: float, dominant: str = "right",
                    use_wrist: bool = False) -> np.ndarray:
    """سرعة الحركة بوحدة (طول الجذع / ثانية) لمركز الجسم، مع إضافة المعصم عند الحاجة."""
    center = (xy[:, LM["l_hip"]] + xy[:, LM["r_hip"]]) / 2
    v = np.gradient(center, axis=0) * fps / torso
    speed = np.hypot(v[:, 0], v[:, 1])
    if use_wrist:
        wr = xy[:, _side(dominant)[2]]
        wv = np.gradient(wr, axis=0) * fps / torso
        speed = np.maximum(speed, 0.5 * np.hypot(wv[:, 0], wv[:, 1]))
    return _moving_average(speed, 3)


def detect_onset(speed: np.ndarray, fps: float, stim_f: int, end_f: int,
                 baseline_s: float = 0.4, k: float = 4.0, min_abs: float = 0.4,
                 sustain_s: float = 0.025):
    """أول إطار تبدأ فيه الحركة بعد المثير.

    الحد = max(min_abs, متوسط الأساس + k × الانحراف المعياري) حيث الأساس هو آخر
    baseline_s ثانية قبل المثير. يعيد (onset_frame | None, threshold, pre_moving).
    """
    T = len(speed)
    b0 = max(0, stim_f - int(baseline_s * fps))
    base = speed[b0:stim_f] if stim_f - b0 >= 3 else speed[: max(3, min(T, stim_f + 1))]
    mu, sd = float(np.mean(base)), float(np.std(base))
    thr = max(min_abs, mu + k * sd)
    pre_moving = mu > 0.6 * min_abs
    sustain = max(2, int(round(sustain_s * fps)))
    end_f = min(end_f, T)
    above = speed > thr
    i = stim_f
    while i + sustain <= end_f:
        if above[i:i + sustain].all():
            j = i
            soft = mu + 2 * sd
            while j > stim_f and speed[j - 1] > soft:
                j -= 1
            return j, thr, pre_moving
        i += 1
    return None, thr, pre_moving


def posture_at(xy: np.ndarray, frame: int, dominant: str = "right") -> dict:
    sh, el, wr = _side(dominant)
    hip = LM["r_hip"] if dominant == "right" else LM["l_hip"]
    f = int(np.clip(frame, 0, len(xy) - 1))
    p = xy[f]
    torso = torso_length(xy)
    mid_sh = (p[LM["l_sh"]] + p[LM["r_sh"]]) / 2
    mid_hip = (p[LM["l_hip"]] + p[LM["r_hip"]]) / 2
    trunk_vec = mid_sh - mid_hip                      # y للأسفل في الصور
    trunk = math.degrees(math.atan2(abs(trunk_vec[0]), max(1e-9, -trunk_vec[1])))
    knees = [angle_deg(p[LM["l_hip"]], p[LM["l_kn"]], p[LM["l_an"]]),
             angle_deg(p[LM["r_hip"]], p[LM["r_kn"]], p[LM["r_an"]])]
    return dict(
        elbow=angle_deg(p[sh], p[el], p[wr]),
        shoulder=angle_deg(p[hip], p[sh], p[el]),
        knee=float(np.nanmin(knees)),            # الركبة الأمامية (الأعمق ثنياً) في طعنة الشبكة
        trunk=trunk,
        reach=float(np.linalg.norm(p[wr] - (p[LM["l_hip"]] + p[LM["r_hip"]]) / 2) / torso),
        hit_height=float((p[hip][1] - p[wr][1]) / torso),
    )


def analyze_trials(xy: np.ndarray, fps: float, onsets_s: list[float], dominant: str = "right",
                   use_wrist: bool = False, contact: bool = False, window_s: float = 2.0,
                   contact_mode: str = "reach") -> list[dict]:
    """يحلّل كل مثير: زمن الاستجابة، السرعة القصوى، والوضع الحركي عند الضرب.

    contact_mode = "reach": لحظة الضرب = أقصى امتداد للمعصم أمام الجسم (ضربات الشبكة)
    contact_mode = "high" : لحظة الضرب = أعلى نقطة للمعصم (ضربات علوية)
    """
    T = len(xy)
    torso = torso_length(xy)
    speed = movement_signal(xy, fps, torso, dominant, use_wrist)
    wr_idx = _side(dominant)[2]
    out = []
    for i, t0 in enumerate(onsets_s):
        sf = int(round(t0 * fps))
        row = dict(idx=i + 1, t_stim=float(t0), rt=np.nan, valid=0, flag="ok", onset_frame=-1,
                   contact_frame=-1, elbow=np.nan, shoulder=np.nan, knee=np.nan,
                   trunk=np.nan, reach=np.nan, hit_height=np.nan, speed_peak=np.nan)
        if sf < 0 or sf >= T - 3:
            row["flag"] = "out_of_video"
            out.append(row)
            continue
        ef = min(T, sf + int(window_s * fps))
        if i + 1 < len(onsets_s):
            ef = min(ef, int(round(onsets_s[i + 1] * fps)))
        onset, thr, pre_moving = detect_onset(speed, fps, sf, ef)
        if onset is None:
            row["flag"] = "no_response"
            out.append(row)
            continue
        rt = (onset - sf) / fps
        row.update(rt=rt, onset_frame=int(onset))
        row["speed_peak"] = float(np.max(speed[onset:min(T, onset + int(0.5 * fps) + 1)]))
        if pre_moving:
            row["flag"] = "unsteady"
        elif rt < RT_MIN_VALID:
            row["flag"] = "anticipation"
        elif rt > RT_MAX_VALID:
            row["flag"] = "slow"
        else:
            row["valid"] = 1
        if contact:
            if ef > onset:
                if contact_mode == "high":
                    cf = onset + int(np.argmin(xy[onset:ef, wr_idx, 1]))          # أعلى نقطة للمعصم
                else:
                    center = (xy[onset:ef, LM["l_hip"]] + xy[onset:ef, LM["r_hip"]]) / 2
                    cf = onset + int(np.argmax(np.linalg.norm(xy[onset:ef, wr_idx] - center, axis=1)))
                row["contact_frame"] = int(cf)
                row.update(posture_at(xy, cf, dominant))
        out.append(row)
    return out


# ------------------------------------------------- مثيرات من سطوع الفيديو (وميض)
def detect_onsets_from_brightness(series, fps: float, k: float = 6.0, min_gap_s: float = 0.8) -> list[float]:
    """يكتشف لحظات ظهور المثير كقفزات سطوع في منطقة من الفيديو."""
    s = np.asarray(series, dtype=float)
    if len(s) < 5 or np.ptp(s) < 1e-6:
        return []
    d = np.diff(s, prepend=s[0])
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-6
    thr = max(med + k * mad, 0.15 * np.ptp(s))
    idx = np.where(d > thr)[0]
    out: list[float] = []
    last = -1e9
    for i in idx:
        t = i / fps
        if t - last >= min_gap_s:
            out.append(float(t))
            last = t
    return out


def onsets_from_offsets(first_onset_s: float, offsets_s: list[float]) -> list[float]:
    """أزمنة المثيرات في الفيديو = زمن أول مثير + إزاحات من مولّد المثيرات."""
    if not offsets_s:
        return [float(first_onset_s)]
    base = offsets_s[0]
    return [float(first_onset_s + (o - base)) for o in offsets_s]


def parse_float_list(text: str) -> list[float]:
    vals = []
    for tok in text.replace("؛", ",").replace("،", ",").replace(";", ",").replace("\n", ",").split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    return vals
