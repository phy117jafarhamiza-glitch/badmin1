"""بيانات اصطناعية للاختبار والعرض التجريبي فقط (ليست بيانات حقيقية).

  * synthetic_session: نقاط جسم اصطناعية لشكل بشري مبسط بزمن استجابة معلوم.
  * seed_demo: ملء قاعدة البيانات بلاعبين وهميين لتجربة لوحة الباحث ونماذج الشبكات العصبية العميقة.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import db
from .exercises import UNIT_PLAN, apply_difficulty, get_exercise
from .config import ZONE_SCORES
from .metrics import LM


def _ease(x):
    x = np.clip(x, 0, 1)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def synthetic_session(fps: float = 120.0, rts=(0.25, 0.30, 0.22), first=1.0, gap=3.0,
                      contact: bool = False, noise: float = 0.4, seed: int = 0, dx: float = 70.0):
    """يعيد (kp, onsets_s). kp شكله (T,33,4) بإحداثيات بكسل (الإطار 640×480)."""
    rng = np.random.default_rng(seed)
    onsets = [first + i * gap for i in range(len(rts))]
    dur = onsets[-1] + 2.5
    T = int(dur * fps)
    t = np.arange(T) / fps

    base = np.zeros((33, 2))
    cx = 320.0
    base[LM["nose"]] = (cx, 180)
    base[LM["l_sh"]] = (cx - 25, 220); base[LM["r_sh"]] = (cx + 25, 220)
    base[LM["l_hip"]] = (cx - 15, 300); base[LM["r_hip"]] = (cx + 15, 300)
    base[LM["l_kn"]] = (cx - 15, 360); base[LM["r_kn"]] = (cx + 15, 360)
    base[LM["l_an"]] = (cx - 15, 420); base[LM["r_an"]] = (cx + 15, 420)
    base[LM["l_el"]] = (cx - 30, 265); base[LM["l_wr"]] = (cx - 30, 305)
    rest_el, rest_wr = np.array([cx + 30, 265.0]), np.array([cx + 30, 305.0])
    up_el, up_wr = np.array([cx + 62, 240.0]), np.array([cx + 100, 262.0])   # ذراع ممدودة أماماً (ضربة شبكة)

    kp = np.zeros((T, 33, 4))
    kp[:, :, :2] = base[None]
    kp[:, :, 3] = 1.0
    shift = np.zeros(T)
    arm = np.zeros(T)
    for t0, rt in zip(onsets, rts):
        on = t0 + rt
        shift += dx * _ease((t - on) / 0.45)
        shift -= dx * _ease((t - (on + 1.2)) / 0.45)   # العودة
        if contact:
            arm += _ease((t - on) / 0.35) - _ease((t - (on + 0.55)) / 0.35)
    kp[:, :, 0] += shift[:, None]
    el = rest_el[None] + (up_el - rest_el)[None] * arm[:, None]
    wr = rest_wr[None] + (up_wr - rest_wr)[None] * arm[:, None]
    kp[:, LM["r_el"], 0] = el[:, 0] + shift; kp[:, LM["r_el"], 1] = el[:, 1]
    kp[:, LM["r_wr"], 0] = wr[:, 0] + shift; kp[:, LM["r_wr"], 1] = wr[:, 1]
    kp[:, :, :2] += rng.normal(0, noise, size=kp[:, :, :2].shape)
    return kp, onsets


def brightness_series(fps: float, onsets, dur: float, level_lo=40.0, level_hi=200.0, flash_s=0.4, seed=0):
    rng = np.random.default_rng(seed)
    T = int(dur * fps)
    s = np.full(T, level_lo) + rng.normal(0, 1.0, T)
    for t0 in onsets:
        a, b = int(t0 * fps), int((t0 + flash_s) * fps)
        s[a:b] = level_hi
    return s


# ----------------------------------------------------------------- بيانات تجريبية
def seed_demo(n_per_group: int = 6, seed: int = 1, path=None) -> int:
    """لاعبون وهميون (مجموعات ثلاث) مع 12 وحدة؛ تحسّن أكبر لمجموعات الذكاء الاصطناعي."""
    rng = np.random.default_rng(seed)
    gain = {"cv": 0.20, "dl": 0.26, "ctrl": 0.10}      # نسبة تحسن RT نهاية البرنامج (تجريبية)
    count = 0
    for grp in ("cv", "dl", "ctrl"):
        for i in range(n_per_group):
            pid = f"D-{grp.upper()}{i + 1:02d}"
            if not db.add_player(pid, grp, "right", is_demo=1, path=path):
                continue
            count += 1
            base_rt = rng.normal(0.42, 0.05)
            base_acc = rng.normal(1.8, 0.4)
            skill = rng.normal(0, 1)

            def tests(phase, frac):
                rt = base_rt * (1 - gain[grp] * frac) + rng.normal(0, 0.01)
                db.set_test_result(pid, phase, "reaction", rt, path=path)
                for name in ("net_straight_acc", "net_cross_acc", "net_lift_acc"):
                    acc = base_acc + frac * (gain[grp] * 6) + rng.normal(0, 0.25)
                    db.set_test_result(pid, phase, name, max(0, acc), path=path)
            tests("pre", 0.0)
            tests("post", 1.0)
            if grp == "ctrl":
                continue
            for p in UNIT_PLAN:
                frac = p["unit"] / 12
                for code in p["codes"]:
                    ex = get_exercise(code)
                    trials = []
                    for k in range(ex.reps):
                        lvl_pen = {1: 0, 2: 0.06, 3: 0.04, 4: 0.12}[ex.level]
                        rt = base_rt * (1 - gain[grp] * frac) + lvl_pen + rng.normal(0, 0.04)
                        flag, valid = "ok", 1
                        if rng.random() < 0.05:
                            rt, flag, valid = float(rng.uniform(0.04, 0.09)), "anticipation", 0
                        row = dict(idx=k + 1, t_stim=1 + 3 * k, rt=rt, valid=valid, flag=flag,
                                   speed_peak=float(rng.normal(2.2 + 0.8 * frac, 0.3)))
                        if ex.contact:
                            elbow = float(np.clip(rng.normal(145 + 12 * frac + 4 * skill, 8), 100, 180))
                            row.update(
                                elbow=elbow,
                                shoulder=float(np.clip(rng.normal(70 + 10 * frac, 12), 20, 140)),
                                knee=float(np.clip(rng.normal(135 - 15 * frac, 10), 80, 175)),
                                trunk=float(abs(rng.normal(25 + 5 * frac, 6))),
                                reach=float(np.clip(rng.normal(1.2 + 0.4 * frac + 0.1 * skill, 0.15), 0.6, 2.4)),
                                hit_height=float(rng.normal(0.4 + 0.1 * frac, 0.2)))
                            p_good = np.clip(0.2 + 0.45 * frac + 0.08 * skill + (row["reach"] - 1.2) * 0.5
                                             + (elbow - 150) / 300, 0.05, 0.95)
                            zone = int(rng.choice([0, 1, 2, 3, 4], p=_zone_probs(p_good)))
                            row.update(zone=zone, acc_score=float(ZONE_SCORES[zone]))
                        trials.append(row)
                    db.add_session(pid, code, trials, unit=p["unit"], phase="train", fps=120.0,
                                   source="demo", difficulty=3, path=path)
    return count


def _zone_probs(p_good: float):
    w = np.array([1 - p_good, 0.15, 0.2, 0.25, 0.4 * p_good + 0.05], dtype=float)
    w[0] = max(w[0], 0.02)
    return w / w.sum()
