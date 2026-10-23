"""محرك البث المباشر: يستقبل نقاط الجسم إطاراً بإطار، ويدير المثيرات، ويحسب زمن الاستجابة
والوضع الحركي لحظياً. لا يعتمد على Streamlit ولا على الكاميرا، فيُختبر ببيانات اصطناعية.

ملاحظة علمية: زمن الاستجابة اللحظي = الزمن الحقيقي + تأخير عرض المثير + تأخير التقاط الحركة
(بضع عشرات إلى مئات المللي ثانية حسب الشبكة والجهاز). لذلك يُستعمل للتغذية الراجعة والتدريب،
أما القياس البحثي فالأفضل من «التحليل بعد التمرين» بفيديو عالي الإطارات.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import metrics
from .config import POSTURE_REFERENCE, RT_MAX_VALID, RT_MIN_VALID

COLORS4 = [("أحمر", (40, 40, 220)), ("أخضر", (60, 160, 60)), ("أزرق", (220, 120, 30)), ("أصفر", (30, 200, 240))]
ARROWS = [("يسار", (-1, 0)), ("يمين", (1, 0)), ("أمام", (0, -1)), ("خلف", (0, 1))]
SKILLS = {"net_straight": "شبكة مستقيمة", "net_cross": "شبكة قطرية", "net_lift": "رفع"}


@dataclass
class Stim:
    label: str                 # وصف عربي يظهر في اللوحة
    bg: tuple = (60, 160, 60)  # لون كتلة المثير (BGR)
    arrow: Optional[tuple] = None   # اتجاه سهم يُرسم فوق الكتلة
    digit: Optional[int] = None


def make_stimulus(mode: str, skill: str = "net_straight", fake_prob: float = 0.3, rng: Optional[random.Random] = None) -> Stim:
    rng = rng or random
    if mode == "single_color":
        return Stim("لون أخضر", (60, 160, 60))
    if mode == "four_colors":
        n, c = rng.choice(COLORS4)
        return Stim(n, c)
    if mode in ("arrows", "fake_arrow", "delayed_arrow"):
        n, d = rng.choice(ARROWS)
        return Stim("سهم " + n, (200, 110, 30), arrow=d)
    if mode == "numbers":
        k = rng.randint(1, 6)
        return Stim(f"منطقة {k}", (150, 30, 110), digit=k)
    if mode == "sound":
        return Stim("إشارة (وميض)", (255, 255, 255))
    if mode == "coach_cue":
        return Stim("تحرّك الآن", (60, 160, 60))
    if mode == "skill_cue":
        key = rng.choice(list(SKILLS)) if skill == "mixed" else skill
        z = rng.randint(1, 4)
        arrow = {"net_straight": (0, -1), "net_cross": (1, -1), "net_lift": (0, -1)}[key]
        return Stim(f"{SKILLS[key]} – منطقة {z}", (200, 110, 30), arrow=arrow, digit=z)
    return Stim("مثير", (60, 160, 60))


def posture_status(angles: dict, contact: bool, use_arm: bool) -> dict:
    """يقارن الزوايا اللحظية بالمدى المرجعي: key -> (value, 'ok'|'low'|'high', label)."""
    out = {}
    for key, (lo, hi, label) in POSTURE_REFERENCE.items():
        if key in ("reach", "hit_height"):
            continue
        if key in ("elbow", "shoulder") and not use_arm:
            continue
        v = angles.get(key, float("nan"))
        if not np.isfinite(v):
            continue
        st = "low" if v < lo else ("high" if v > hi else "ok")
        out[key] = (float(v), st, label)
    return out


def frame_angles(xy_frame: np.ndarray, dominant: str) -> dict:
    """زوايا الوضع في إطار واحد (xy_frame: 33×2)."""
    return metrics.posture_at(xy_frame[None, :, :], 0, dominant)


@dataclass
class LiveSession:
    stim_cfg: dict
    n_trials: int = 6
    dominant: str = "right"
    use_wrist: bool = False
    contact: bool = False
    rest_s: float = 2.0
    seed: Optional[int] = None
    # حالة داخلية
    phase: str = "idle"             # idle|countdown|waiting|stimulus|rest|done
    trials: list = field(default_factory=list)
    stim: Optional[Stim] = None
    message: str = "اضغط «ابدأ»"
    angles: dict = field(default_factory=dict)
    status: dict = field(default_factory=dict)
    keyframes: list = field(default_factory=list)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self._t = []          # أزمنة
        self._xy = []         # نقاط (33×2)
        self._speed = []
        self._phase_t0 = 0.0
        self._wait = 0.0
        self._stim_t = 0.0
        self._run = 0         # عدّاد إطارات متتالية فوق الحد
        self._run_start = None
        self._thr = None
        self._pre_moving = False
        self._best = None     # أفضل إطار امتداد بعد الحركة
        self._onset_t = None
        self._fake_done = False
        self.last_rt: Optional[float] = None
        self.detected_frames = 0
        self.total_frames = 0

    # ---------------------------------------------------------------- تحكم
    def start(self, t: float):
        self.phase, self._phase_t0 = "countdown", t
        self.trials, self.keyframes = [], []
        self.message = "استعد"

    def stop(self):
        self.phase = "done"
        self.message = "أُوقفت الجلسة"

    @property
    def done(self) -> bool:
        return self.phase == "done"

    # ---------------------------------------------------------------- معالجة الإطار
    def _torso(self) -> float:
        xs = np.array(self._xy[-90:])
        return metrics.torso_length(xs)

    def _push(self, t: float, xy: np.ndarray):
        self._t.append(t)
        self._xy.append(xy)
        if len(self._t) > 600:
            del self._t[:100], self._xy[:100], self._speed[:100]
        if len(self._t) < 2:
            self._speed.append(0.0)
            return
        dt = max(1e-3, self._t[-1] - self._t[-2])
        tl = self._torso()
        c0 = (self._xy[-2][metrics.LM["l_hip"]] + self._xy[-2][metrics.LM["r_hip"]]) / 2
        c1 = (xy[metrics.LM["l_hip"]] + xy[metrics.LM["r_hip"]]) / 2
        sp = float(np.hypot(*(c1 - c0))) / dt / tl
        if self.use_wrist:
            w = metrics._side(self.dominant)[2]
            sp = max(sp, 0.5 * float(np.hypot(*(xy[w] - self._xy[-2][w]))) / dt / tl)
        prev = self._speed[-1]
        self._speed.append(0.5 * sp + 0.5 * prev)        # تنعيم أسّي خفيف

    def update(self, t: float, xy: Optional[np.ndarray]) -> dict:
        """xy: (33,2) بالبكسل أو None عند عدم اكتشاف اللاعب. يعيد حالة العرض."""
        self.total_frames += 1
        if xy is not None and np.isfinite(xy).all():
            self.detected_frames += 1
            self._push(t, xy)
            self.angles = frame_angles(xy, self.dominant)
            self.status = posture_status(self.angles, self.contact, self.use_wrist or self.contact)
            self._track_reach(t, xy)
        elif self.phase in ("waiting", "stimulus"):
            self.message = "لا يظهر اللاعب في الإطار"
        self._advance(t)
        return self.state(t)

    def _track_reach(self, t: float, xy: np.ndarray):
        if self.phase == "stimulus" and self._onset_t is not None and self.contact:
            w = metrics._side(self.dominant)[2]
            hip = (xy[metrics.LM["l_hip"]] + xy[metrics.LM["r_hip"]]) / 2
            d = float(np.hypot(*(xy[w] - hip)))
            if self._best is None or d > self._best[0]:
                self._best = (d, xy.copy(), t)

    def _advance(self, t: float):
        ph = self.phase
        if ph == "countdown":
            left = 3 - (t - self._phase_t0)
            self.message = f"ابدأ بعد {max(0, math.ceil(left))}"
            if left <= 0:
                self._begin_wait(t)
        elif ph == "waiting":
            self.message = "استعد… ثبّت وضع الاستعداد"
            if t - self._phase_t0 >= self._wait:
                self._show_stim(t)
        elif ph == "stimulus":
            self._watch(t)
        elif ph == "rest":
            if t - self._phase_t0 >= self.rest_s:
                if len(self.trials) >= self.n_trials:
                    self.phase, self.message = "done", "انتهت المجموعة"
                else:
                    self._begin_wait(t)

    def _begin_wait(self, t: float):
        self.phase, self._phase_t0 = "waiting", t
        self._wait = self.rng.uniform(self.stim_cfg.get("delay_min", 1.0), self.stim_cfg.get("delay_max", 3.0))
        self.stim = None

    def _show_stim(self, t: float):
        mode = self.stim_cfg.get("mode", "single_color")
        self.stim = make_stimulus(mode, self.stim_cfg.get("skill", "net_straight"), self.stim_cfg.get("fake_prob", 0.3), self.rng)
        self.phase, self._stim_t, self._phase_t0 = "stimulus", t, t
        self.message = "الآن! " + self.stim.label
        # أساس الحركة قبل المثير (آخر 0.4 ثانية)
        idx = [i for i, tt in enumerate(self._t) if t - 0.4 <= tt <= t]
        base = np.array([self._speed[i] for i in idx]) if len(idx) >= 3 else np.array([0.0])
        mu, sd = float(base.mean()), float(base.std())
        self._thr = max(0.4, mu + 4 * sd)
        self._pre_moving = mu > 0.6 * 0.4
        self._run, self._run_start, self._onset_t, self._best = 0, None, None, None

    def _watch(self, t: float):
        if self._onset_t is None:
            if self._speed and self._speed[-1] > self._thr:
                self._run += 1
                if self._run_start is None:
                    self._run_start = self._t[-1]
                if self._run >= 3:
                    self._onset_t = self._run_start
            else:
                self._run, self._run_start = 0, None
            if self._onset_t is None and t - self._stim_t > RT_MAX_VALID:
                self._finish(t, None)
            elif self._onset_t is not None:
                if not self.contact:
                    self._finish(t, self._onset_t)
        elif self.contact and t - self._onset_t > 1.0:
            self._finish(t, self._onset_t)

    def _finish(self, t: float, onset_t: Optional[float]):
        n = len(self.trials) + 1
        row = dict(idx=n, t_stim=self._stim_t, rt=float("nan"), valid=0, flag="no_response", onset_frame=-1,
                   contact_frame=-1, speed_peak=float(max(self._speed[-30:], default=0.0)))
        if onset_t is not None:
            rt = onset_t - self._stim_t
            row["rt"] = rt
            if self._pre_moving:
                row["flag"] = "unsteady"
            elif rt < RT_MIN_VALID:
                row["flag"] = "anticipation"
            elif rt > RT_MAX_VALID:
                row["flag"] = "slow"
            else:
                row["flag"], row["valid"] = "ok", 1
            self.last_rt = rt
        if self.contact and self._best is not None:
            _, bxy, _bt = self._best
            row.update({k: v for k, v in metrics.posture_at(bxy[None], 0, self.dominant).items() if k in (
                "elbow", "shoulder", "knee", "trunk", "reach", "hit_height")})
            row["_xy"] = bxy
        self.trials.append(row)
        self.phase, self._phase_t0 = "rest", t
        self.message = (f"زمن الاستجابة {row['rt'] * 1000:.0f} م.ث" if np.isfinite(row["rt"]) else "لا استجابة")

    # ---------------------------------------------------------------- مخرجات
    def state(self, t: float) -> dict:
        return dict(phase=self.phase, message=self.message, stim=self.stim, last_rt=self.last_rt,
                    n_done=len(self.trials), n_trials=self.n_trials, status=self.status,
                    angles=self.angles, detected=self.detected_ratio)

    @property
    def detected_ratio(self) -> float:
        return self.detected_frames / self.total_frames if self.total_frames else 0.0

    def result_rows(self) -> list[dict]:
        """صفوف المحاولات بالصيغة التي تحفظها قاعدة البيانات (بلا بيانات داخلية)."""
        return [{k: v for k, v in r.items() if k != "_xy"} for r in self.trials]


# ====================================================================== الرسم على الإطار
STATUS_COLOR = {"ok": (60, 180, 60), "low": (30, 120, 240), "high": (30, 60, 220)}   # BGR
JOINT_OF = {"elbow": "el", "shoulder": "sh", "knee": "kn"}


def _joint_index(key: str, dominant: str, xy: np.ndarray) -> int:
    side = "r" if dominant == "right" else "l"
    if key == "elbow":
        return metrics.LM[f"{side}_el"]
    if key == "shoulder":
        return metrics.LM[f"{side}_sh"]
    if key == "knee":      # الركبة الأعمق ثنياً
        a = [metrics.angle_deg(xy[metrics.LM[f"{x}_hip"]], xy[metrics.LM[f"{x}_kn"]], xy[metrics.LM[f"{x}_an"]]) for x in ("l", "r")]
        return metrics.LM["l_kn"] if np.nanargmin(a) == 0 else metrics.LM["r_kn"]
    return metrics.LM[f"{side}_sh"]


def draw_stimulus(img, stim: Optional[Stim], phase: str, size: int = 150):
    import cv2
    h, w = img.shape[:2]
    x0, y0 = 10, 10
    if stim is None or phase != "stimulus":
        cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (90, 90, 90), -1)
        cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 2)
        return img
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), stim.bg, -1)
    c = (x0 + size // 2, y0 + size // 2)
    if stim.arrow is not None:
        d = np.array(stim.arrow, float)
        d /= np.linalg.norm(d)
        a, b = np.array(c) - d * size * 0.32, np.array(c) + d * size * 0.32
        cv2.arrowedLine(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), 8, tipLength=0.4)
    if stim.digit is not None:
        cv2.putText(img, str(stim.digit), (x0 + size - 48, y0 + size - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (255, 255, 255), 4, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), stim.bg, 8)          # إطار ملوّن يلفت الانتباه
    return img


def render_frame(img_bgr, xy: Optional[np.ndarray], state: dict, dominant: str = "right", ref_bgr=None):
    """يرسم الهيكل والمثير والأرقام على إطار البث، ويلصق الصورة المرجعية جانبه."""
    import cv2
    from .pose import draw_skeleton
    from .reference import side_by_side
    img = img_bgr.copy()
    if xy is not None:
        labels = {}
        img = draw_skeleton(img, xy, 1.0, None)
        for key, (val, st, _lab) in state.get("status", {}).items():
            j = _joint_index(key, dominant, xy)
            p = tuple(int(v) for v in xy[j])
            cv2.circle(img, p, 9, STATUS_COLOR[st], -1, cv2.LINE_AA)
            cv2.putText(img, f"{val:.0f}", (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, f"{val:.0f}", (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, STATUS_COLOR[st], 2, cv2.LINE_AA)
    img = draw_stimulus(img, state.get("stim"), state.get("phase", "idle"))
    h, w = img.shape[:2]
    if state.get("last_rt") is not None:
        txt = f"{state['last_rt'] * 1000:.0f} ms"
        cv2.putText(img, txt, (w - 190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(img, txt, (w - 190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, f"{state.get('n_done', 0)}/{state.get('n_trials', 0)}", (w - 110, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
    if ref_bgr is not None:
        img = side_by_side(img, ref_bgr)
    return img


def pose_card(xy: np.ndarray, ref_bgr, dominant: str = "right", status: Optional[dict] = None):
    """بطاقة مقارنة: هيكل اللاعب في لحظة الضربة (يسار) والوضع الصحيح (يمين)."""
    import cv2
    from .pose import draw_skeleton
    from .reference import side_by_side
    h, w = ref_bgr.shape[:2]
    canvas = np.full((h, w, 3), 245, np.uint8)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    sc = 0.8 * min(w, h) / span
    shifted = (xy - (lo + hi) / 2) * sc + np.array([w / 2, h / 2])
    canvas = draw_skeleton(canvas, shifted, 1.0, None)
    if status:
        for key, (val, st, _l) in status.items():
            j = _joint_index(key, dominant, xy)
            p = tuple(int(v) for v in shifted[j])
            cv2.circle(canvas, p, 8, STATUS_COLOR[st], -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{val:.0f}", (p[0] + 9, p[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, STATUS_COLOR[st], 2, cv2.LINE_AA)
    return side_by_side(canvas, ref_bgr)
