"""الصورة المرجعية للأداء الصحيح.

ترسم مخططاً هيكلياً جانبياً من زوايا المدى المرجعي (POSTURE_REFERENCE) — وهو رسم توضيحي
مبني على قيم مبدئية يعتمدها الخبراء، وليس صورة فوتوغرافية. إن وجدت صورة حقيقية باسم
`assets/reference/<kind>.png` (أو `.jpg`) فتُستخدم بدلاً من الرسم، فيمكنك وضع صور لاعب
نموذجي معتمد من الخبراء دون تغيير الكود.

لا نصوص عربية داخل الصورة (OpenCV لا يشكّل الحروف العربية)؛ تُعرض الشروح في الواجهة.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import POSTURE_REFERENCE

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "reference"

# الزوايا المرجعية لكل وضع: منتصف المدى الوارد في الإعدادات إلا وضع الاستعداد
KINDS = {
    "ready": dict(trunk=12.0, knee=140.0, elbow=100.0, shoulder=35.0, front_thigh=18.0, back_thigh=-14.0,
                  back_knee=145.0, racket=False),
    "net": dict(trunk=None, knee=None, elbow=None, shoulder=None, front_thigh=62.0, back_thigh=-22.0,
                back_knee=172.0, racket=True),
}

COLORS = dict(bg=(245, 245, 245), body=(60, 60, 60), good=(60, 160, 60), front=(200, 100, 20),
              racket=(30, 30, 200), floor=(190, 190, 190))


def _mid(key: str) -> float:
    lo, hi, _ = POSTURE_REFERENCE[key]
    return (lo + hi) / 2


def target_angles(kind: str) -> dict:
    spec = dict(KINDS[kind])
    for k in ("trunk", "knee", "elbow", "shoulder"):
        if spec.get(k) is None:
            spec[k] = _mid(k)
    return spec


def _rot(v, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _limb(base, toward, ang, length, pick):
    """نقطة نهاية طرف يصنع عند base زاوية ang مع الاتجاه toward؛ pick يختار أحد الحلّين."""
    cands = [base + _unit(_rot(_unit(toward), s * ang)) * length for s in (1, -1)]
    return cands[0] if pick(cands[0]) >= pick(cands[1]) else cands[1]


def skeleton_points(spec: dict, w: int, h: int) -> dict:
    """نقاط الجسم (إحداثيات صورة، يمين = أمام اللاعب)."""
    s = h / 480.0
    Lt, Lu, Lf, Lth, Lsh = 120 * s, 56 * s, 56 * s, 82 * s, 82 * s
    hip = np.array([w * 0.27, h * 0.52])
    t = math.radians(spec["trunk"])
    sh = hip + Lt * np.array([math.sin(t), -math.cos(t)])
    head = sh + 34 * s * _unit(sh - hip)
    # الذراع: زاوية الكتف بين الجذع والذراع العلوية
    el = _limb(sh, hip - sh, spec["shoulder"], Lu, lambda p: p[0])
    wr = _limb(el, sh - el, spec["elbow"], Lf, lambda p: p[0] - 0.3 * p[1] if spec["racket"] else p[0])
    # الساق الأمامية والخلفية
    kn_f = hip + Lth * np.array([math.sin(math.radians(spec["front_thigh"])), math.cos(math.radians(spec["front_thigh"]))])
    an_f = _limb(kn_f, hip - kn_f, spec["knee"], Lsh, lambda p: p[1] - 0.1 * p[0])
    kn_b = hip + Lth * np.array([math.sin(math.radians(spec["back_thigh"])), math.cos(math.radians(spec["back_thigh"]))])
    an_b = _limb(kn_b, hip - kn_b, spec["back_knee"], Lsh, lambda p: p[1] - 0.1 * (-p[0]))
    return dict(hip=hip, sh=sh, head=head, el=el, wr=wr, kn_f=kn_f, an_f=an_f, kn_b=kn_b, an_b=an_b,
                floor_y=max(an_f[1], an_b[1]) + 6 * s, scale=s)


def draw_reference(kind: str = "net", w: int = 320, h: int = 420, with_racket: bool = True):
    """يعيد صورة BGR للوضع الصحيح (صورة معتمدة من مجلد assets إن وُجدت)."""
    import cv2
    for ext in ("png", "jpg", "jpeg"):
        f = ASSETS / f"{kind}.{ext}"
        if f.exists():
            img = cv2.imread(str(f))
            if img is not None:
                return cv2.resize(img, (w, h))
    spec = target_angles(kind)
    pts = skeleton_points(spec, w, h)
    s = pts["scale"]
    img = np.full((h, w, 3), COLORS["bg"], np.uint8)
    cv2.line(img, (int(w * 0.05), int(pts["floor_y"])), (int(w * 0.95), int(pts["floor_y"])), COLORS["floor"], 2)

    def line(a, b, col=COLORS["body"], th=5):
        cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]), col, int(th * s) + 1, cv2.LINE_AA)

    line("an_b", "kn_b"); line("kn_b", "hip")
    line("hip", "kn_f", COLORS["front"]); line("kn_f", "an_f", COLORS["front"])
    line("hip", "sh"); line("sh", "el", COLORS["good"]); line("el", "wr", COLORS["good"])
    cv2.circle(img, tuple(int(v) for v in pts["head"]), int(17 * s), COLORS["body"], 3, cv2.LINE_AA)
    for k in ("sh", "el", "wr", "hip", "kn_f", "an_f", "kn_b", "an_b"):
        cv2.circle(img, tuple(int(v) for v in pts[k]), int(5 * s), (0, 0, 0), -1, cv2.LINE_AA)
    if with_racket and spec["racket"]:
        d = _unit(pts["wr"] - pts["el"])
        tip = pts["wr"] + d * 62 * s
        cv2.line(img, tuple(int(v) for v in pts["wr"]), tuple(int(v) for v in tip), COLORS["racket"], int(3 * s) + 1, cv2.LINE_AA)
        cv2.circle(img, tuple(int(v) for v in tip + d * 14 * s), int(15 * s), COLORS["racket"], 2, cv2.LINE_AA)
    # أرقام الزوايا المستهدفة عند المفاصل
    font = cv2.FONT_HERSHEY_SIMPLEX
    for k, joint in (("elbow", "el"), ("shoulder", "sh"), ("knee", "kn_f")):
        off = np.array([-44 * s, 4 * s]) if k == "shoulder" else np.array([8 * s, -8 * s])
        p = pts[joint] + off
        cv2.putText(img, f"{spec[k]:.0f}", (int(p[0]), int(p[1])), font, 0.55 * s + 0.15, (0, 120, 0), 2, cv2.LINE_AA)
    return img


def side_by_side(left_bgr, right_bgr, gap: int = 8):
    """يضع صورتين متجاورتين بنفس الارتفاع (يمين الشاشة = المرجع)."""
    import cv2
    h = left_bgr.shape[0]
    r = right_bgr
    if r.shape[0] != h:
        r = cv2.resize(r, (int(r.shape[1] * h / r.shape[0]), h))
    sep = np.full((h, gap, 3), 255, np.uint8)
    return np.hstack([left_bgr, sep, r])


def kind_for(contact: bool) -> str:
    return "net" if contact else "ready"
