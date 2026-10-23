"""التغذية الراجعة النصية (مجموعة الرؤية الحاسوبية) بالعربية."""
from __future__ import annotations

import numpy as np

from .config import POSTURE_REFERENCE

FLAG_AR = {
    "ok": "صالحة",
    "anticipation": "استباق (تحرك قبل المثير)",
    "no_response": "لا استجابة",
    "slow": "بطيئة جداً",
    "unsteady": "حركة قبل المثير",
    "out_of_video": "خارج الفيديو",
}


def rt_feedback(summary: dict, previous: dict | None) -> list[str]:
    msgs = []
    if not summary or not np.isfinite(summary.get("mean_rt", np.nan)):
        return ["لا توجد محاولات صالحة لحساب زمن الاستجابة."]
    msgs.append(f"متوسط زمن الاستجابة {summary['mean_rt'] * 1000:.0f} مللي ثانية، "
                f"وأفضل محاولة {summary['best_rt'] * 1000:.0f} مللي ثانية.")
    if previous and np.isfinite(previous.get("mean_rt", np.nan)):
        d = previous["mean_rt"] - summary["mean_rt"]
        if abs(d) < 0.01:
            msgs.append("الزمن قريب جداً من الجلسة السابقة.")
        elif d > 0:
            msgs.append(f"تحسّن بمقدار {d * 1000:.0f} مللي ثانية عن الجلسة السابقة.")
        else:
            msgs.append(f"تراجع بمقدار {-d * 1000:.0f} مللي ثانية عن الجلسة السابقة؛ ركّز على الاستعداد قبل الإشارة.")
    if summary.get("sd_rt") and np.isfinite(summary["sd_rt"]) and summary["sd_rt"] > 0.08:
        msgs.append("التباين بين المحاولات مرتفع؛ حافظ على وضع استعداد ثابت قبل كل مثير.")
    if summary.get("anticipation_rate", 0) > 0.15:
        msgs.append("نسبة الاستباق مرتفعة؛ انتظر ظهور المثير ولا تخمّن توقيته.")
    return msgs


def posture_feedback(trials: list[dict]) -> list[str]:
    msgs = []
    rows = [t for t in trials if t.get("valid") and np.isfinite(t.get("elbow", np.nan))]
    if not rows:
        return msgs
    for key, (lo, hi, label) in POSTURE_REFERENCE.items():
        vals = [t[key] for t in rows if np.isfinite(t.get(key, np.nan))]
        if not vals:
            continue
        m = float(np.mean(vals))
        if m < lo:
            msgs.append(f"{label}: المتوسط {m:.0f} أقل من المدى المرجعي ({lo:.1f}–{hi:.1f}).")
        elif m > hi:
            msgs.append(f"{label}: المتوسط {m:.0f} أعلى من المدى المرجعي ({lo:.1f}–{hi:.1f}).")
        else:
            msgs.append(f"{label}: المتوسط {m:.0f} ضمن المدى المرجعي ({lo:.1f}–{hi:.1f}).")
    return msgs
