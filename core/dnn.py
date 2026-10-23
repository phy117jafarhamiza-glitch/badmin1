"""وحدة الشبكات العصبية العميقة (المجموعة التجريبية الثانية): نموذج عميق، ملف نقاط الضعف، الصعوبة، الوصفة.

ما هو «شبكة عصبية عميقة» وما هو «قواعد» في هذه الوحدة (للإفصاح العلمي في الأطروحة):
  * شبكة عصبية عميقة متعددة الطبقات (3 طبقات مخفية) تتنبأ بدرجة دقة ضربة الشبكة من المؤشرات الحركية
    -> تعلّم عميق يُدرَّب على بيانات الدراسة، ويُقارَن بنموذجين مرجعيين (الغابة العشوائية والانحدار الخطي).
  * تحليل «ماذا لو» بالشبكة العميقة: أي مؤشر حركي يرفع الدقة المتوقعة أكثر إن اقترب من قيمة أفضل المحاولات.
  * ملف نقاط الضعف (z-score مقابل بيانات المجموعة)                -> إحصاء وصفي
  * ضبط الصعوبة واختيار التمرين                                  -> قواعد صريحة موثّقة (تُضبط في التجربة الاستطلاعية)
  * ملاحظة: نموذج تقدير الوضعية (BlazePose في MediaPipe) نفسه شبكة عصبية التفافية، ويُستخدم في المجموعتين.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import POSTURE_REFERENCE
from .exercises import (DIFFICULTY_DEFAULT, DIFFICULTY_MAX, DIFFICULTY_MIN, SKILL_LABELS,
                        apply_difficulty, exercises_for_unit)

FEATURES = ["rt", "elbow", "shoulder", "knee", "trunk", "reach", "hit_height", "speed_peak"]
POSTURE_FEATURES = ["elbow", "shoulder", "knee", "trunk", "reach", "hit_height"]
MIN_TRIALS_FOR_MODEL = 60
HIDDEN_LAYERS = (64, 32, 16)

FEATURE_AR = {
    "rt": "زمن الاستجابة", "elbow": "زاوية المرفق", "shoulder": "زاوية الكتف", "knee": "زاوية الركبة",
    "trunk": "ميل الجذع", "reach": "مدى امتداد الذراع", "hit_height": "ارتفاع المعصم", "speed_peak": "السرعة القصوى",
}


# ---------------------------------------------------------------- ملخصات
def valid_trials(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["valid"] == 1) & df["rt"].notna()] if not df.empty else df


def session_summary(df: pd.DataFrame) -> dict:
    """ملخص جلسة أو مجموعة محاولات."""
    if df.empty:
        return {}
    v = valid_trials(df)
    return dict(
        n=len(df), n_valid=len(v),
        mean_rt=float(v["rt"].mean()) if len(v) else np.nan,
        best_rt=float(v["rt"].min()) if len(v) else np.nan,
        sd_rt=float(v["rt"].std()) if len(v) > 1 else np.nan,
        anticipation_rate=float((df["flag"] == "anticipation").mean()),
        miss_rate=float(df["flag"].isin(["no_response", "slow", "unsteady"]).mean()),
        mean_acc=float(df["acc_score"].mean()) if df["acc_score"].notna().any() else np.nan,
    )


def per_session_series(df: pd.DataFrame) -> pd.DataFrame:
    """متوسط زمن الاستجابة والدقة لكل جلسة بالترتيب الزمني."""
    if df.empty:
        return pd.DataFrame(columns=["session_id", "unit", "mean_rt", "mean_acc", "n_valid"])
    rows = []
    for sid, g in df.groupby("session_id", sort=False):
        s = session_summary(g)
        rows.append(dict(session_id=sid, unit=g["unit"].iloc[0], exercise=g["exercise"].iloc[0],
                         created=g["created"].iloc[0], mean_rt=s["mean_rt"], mean_acc=s["mean_acc"],
                         n_valid=s["n_valid"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------- ملف نقاط الضعف
def weakness_profile(player_df: pd.DataFrame, group_df: pd.DataFrame) -> dict[str, float]:
    """درجة قصور (z) لكل مهارة: القيمة الأعلى = قصور أكبر. تقارن اللاعب بمتوسط المجموعة."""
    prof: dict[str, float] = {}
    pv, gv = valid_trials(player_df), valid_trials(group_df)

    def z_high_bad(p, g):
        if len(p) == 0 or len(g) < 5 or np.nanstd(g) < 1e-9:
            return 0.0
        return float((np.nanmean(p) - np.nanmean(g)) / np.nanstd(g))

    for lvl, key in ((1, "speed"), (2, "choice"), (3, "anticipation")):
        p = pv.loc[pv["level"] == lvl, "rt"] if len(pv) else []
        g = gv.loc[gv["level"] == lvl, "rt"] if len(gv) else []
        prof[key] = z_high_bad(np.asarray(p, float), np.asarray(g, float))
    if len(player_df):
        prof["anticipation"] += 1.5 * float((player_df["flag"] == "anticipation").mean())
    if player_df["acc_score"].notna().any() and group_df["acc_score"].notna().sum() >= 5:
        sd = group_df["acc_score"].std() or 1.0
        prof["accuracy"] = float((group_df["acc_score"].mean() - player_df["acc_score"].mean()) / sd)
    else:
        prof["accuracy"] = 0.0
    dev = []
    for k, (lo, hi, _lbl) in POSTURE_REFERENCE.items():
        if k in player_df and player_df[k].notna().any():
            m = player_df[k].mean()
            span = max(1e-9, hi - lo)
            dev.append(max(0.0, (lo - m) / span, (m - hi) / span))
    prof["technique"] = float(np.mean(dev) * 4) if dev else 0.0
    return prof


def top_weakness(profile: dict[str, float]) -> Optional[str]:
    if not profile:
        return None
    k, v = max(profile.items(), key=lambda kv: kv[1])
    return k if v > 0.25 else None


# ---------------------------------------------------------------- ضبط الصعوبة
def next_difficulty(prev: int, last: dict, before: Optional[dict]) -> tuple[int, str]:
    """قاعدة ضبط الصعوبة: ارتفاع عند تحسن زمن الاستجابة (≥ 5%) مع قلة الأخطاء، وانخفاض عند التراجع."""
    prev = int(prev or DIFFICULTY_DEFAULT)
    if not last or not before or not np.isfinite(last.get("mean_rt", np.nan)) or not np.isfinite(before.get("mean_rt", np.nan)):
        return prev, "لا توجد بيانات سابقة كافية؛ تبقى الصعوبة كما هي."
    change = (before["mean_rt"] - last["mean_rt"]) / before["mean_rt"]
    errors = last.get("anticipation_rate", 0) + last.get("miss_rate", 0)
    if change >= 0.05 and errors <= 0.2:
        return min(DIFFICULTY_MAX, prev + 1), f"تحسن زمن الاستجابة {change:.0%} مع أخطاء قليلة؛ ترتفع الصعوبة."
    if change <= -0.05 or errors > 0.35:
        return max(DIFFICULTY_MIN, prev - 1), "تراجع الأداء أو كثرة الأخطاء؛ تنخفض الصعوبة."
    return prev, "الأداء مستقر؛ تبقى الصعوبة كما هي."


# ---------------------------------------------------------------- وصفة التمرين
def prescribe(unit: int, player_df: pd.DataFrame, group_df: pd.DataFrame, last_difficulty: Optional[int]) -> dict:
    """وصفة الوحدة لمجموعة الشبكات العصبية العميقة: ترتيب التمرينات حسب نقطة الضعف وضبط الصعوبة فردياً."""
    series = per_session_series(player_df)
    last = before = None
    if len(series) >= 1:
        last = dict(series.iloc[-1])
        last.update(session_summary(player_df[player_df["session_id"] == series.iloc[-1]["session_id"]]))
    if len(series) >= 2:
        before = dict(series.iloc[-2])
        before.update(session_summary(player_df[player_df["session_id"] == series.iloc[-2]["session_id"]]))
    diff, why = next_difficulty(last_difficulty or DIFFICULTY_DEFAULT, last or {}, before)

    prof = weakness_profile(player_df, group_df) if len(player_df) else {}
    weak = top_weakness(prof)
    exs = exercises_for_unit(unit)

    def priority(e):
        return -(1 if weak in e.skills else 0), exs.index(e)
    ordered = sorted(exs, key=priority)
    items = []
    for e in ordered:
        item = apply_difficulty(e, diff)
        item["reason"] = ("يستهدف نقطة الضعف: " + SKILL_LABELS[weak]) if weak in e.skills else "ضمن خطة الوحدة"
        items.append(item)
    return dict(difficulty=diff, difficulty_reason=why, weakness=weak, profile=prof, items=items)


# ----------------------------------------------------------------- النماذج
@dataclass
class DeepAccuracyModel:
    model: object              # Pipeline(StandardScaler -> MLPRegressor 3 طبقات مخفية)
    cv_r2: float               # التحقق المتقاطع حسب اللاعب (R²) للشبكة العميقة
    baselines: dict            # R² للنماذج المرجعية: rf, linear
    importances: dict          # أهمية المتغيرات (Permutation importance) مجموعها 1
    n: int
    train_df: pd.DataFrame

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.model.predict(rows[FEATURES])


def _make_dnn(seed: int = 0):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, activation="relu", solver="adam", alpha=1e-3,
                     early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                     max_iter=2000, random_state=seed))


def fit_accuracy_model(df: pd.DataFrame, seed: int = 0) -> tuple[Optional[DeepAccuracyModel], str]:
    """شبكة عصبية عميقة للتنبؤ بدرجة دقة ضربة الشبكة من المؤشرات الحركية (محاولات الضرب فقط)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if df.empty:
        return None, "لا توجد بيانات."
    d = df[(df["acc_score"].notna()) & (df["valid"] == 1)].dropna(subset=FEATURES).copy()
    if len(d) < MIN_TRIALS_FOR_MODEL:
        return None, f"البيانات غير كافية لتدريب الشبكة العميقة ({len(d)} من {MIN_TRIALS_FOR_MODEL} محاولة)."
    X, y, groups = d[FEATURES], d["acc_score"], d["player_id"]
    n_splits = int(min(5, groups.nunique()))
    cv_dnn, base = np.nan, {"rf": np.nan, "linear": np.nan}
    if n_splits >= 2:
        kf = GroupKFold(n_splits=n_splits)
        cv_dnn = float(np.mean(cross_val_score(_make_dnn(seed), X, y, cv=kf, groups=groups, scoring="r2")))
        base["rf"] = float(np.mean(cross_val_score(
            RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=seed), X, y,
            cv=kf, groups=groups, scoring="r2")))
        base["linear"] = float(np.mean(cross_val_score(
            make_pipeline(StandardScaler(), Ridge(alpha=1.0)), X, y, cv=kf, groups=groups, scoring="r2")))
    model = _make_dnn(seed).fit(X, y)
    pi = permutation_importance(model, X, y, n_repeats=8, random_state=seed, scoring="r2")
    imp = np.clip(pi.importances_mean, 0, None)
    imp = imp / imp.sum() if imp.sum() > 0 else np.full(len(FEATURES), 1.0 / len(FEATURES))
    importances = dict(sorted(zip(FEATURES, map(float, imp)), key=lambda kv: -kv[1]))
    return DeepAccuracyModel(model, cv_dnn, base, importances, len(d), d), "تم تدريب الشبكة العصبية العميقة."


def what_if(model: DeepAccuracyModel, player_df: pd.DataFrame, top: int = 3) -> list[dict]:
    """تحليل «ماذا لو»: لكل مؤشر حركي، كم ترتفع الدقة المتوقعة للاعب إن اقترب مؤشره من أفضل المحاولات؟

    القيمة المستهدفة = متوسط المؤشر في أعلى ربع محاولات من حيث الدقة في بيانات التدريب.
    """
    d = player_df[(player_df["valid"] == 1)].dropna(subset=FEATURES)
    if model is None or d.empty:
        return []
    train = model.train_df
    best = train[train["acc_score"] >= train["acc_score"].quantile(0.75)]
    if best.empty:
        return []
    base_row = d[FEATURES].mean().to_frame().T
    base_pred = float(model.predict(base_row)[0])
    out = []
    for f in POSTURE_FEATURES:
        target = float(best[f].mean())
        mod = base_row.copy()
        mod[f] = target
        delta = float(model.predict(mod)[0]) - base_pred
        out.append(dict(feature=f, label=FEATURE_AR[f], current=float(base_row[f].iloc[0]), target=target, delta=delta))
    out = [o for o in out if o["delta"] > 0.02]
    return sorted(out, key=lambda o: -o["delta"])[:top]


def forecast_rt(series: pd.DataFrame, target_unit: int = 12) -> dict:
    """انحدار خطي لمتوسط زمن الاستجابة عبر الوحدات للتنبؤ بقيمته في نهاية البرنامج."""
    s = series.dropna(subset=["mean_rt", "unit"])
    if s["unit"].nunique() < 3:
        return dict(ok=False, msg="يلزم ثلاث وحدات مختلفة على الأقل للتنبؤ.")
    from sklearn.linear_model import LinearRegression
    u = s.groupby("unit")["mean_rt"].mean().reset_index()
    lr = LinearRegression().fit(u[["unit"]], u["mean_rt"])
    pred = float(lr.predict(pd.DataFrame({"unit": [target_unit]}))[0])
    return dict(ok=True, slope=float(lr.coef_[0]), predicted=pred, r2=float(lr.score(u[["unit"]], u["mean_rt"])))
