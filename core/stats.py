"""فحوص إحصائية سريعة للباحث (التحليل الرسمي يبقى في SPSS كما في الفصل الثالث)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import GROUPS


def wide_tests(tests: pd.DataFrame, test: str) -> pd.DataFrame:
    """لاعب × (pre, post) لاختبار واحد، مع المجموعة."""
    t = tests[tests["test"] == test]
    if t.empty:
        return pd.DataFrame(columns=["player_id", "grp", "pre", "post", "gain"])
    w = t.pivot_table(index=["player_id", "grp"], columns="phase", values="value").reset_index()
    for c in ("pre", "post"):
        if c not in w:
            w[c] = np.nan
    w = w.dropna(subset=["pre", "post"])
    w["gain"] = w["post"] - w["pre"]
    return w


def homogeneity_levene(w: pd.DataFrame) -> dict:
    groups = [g["pre"].values for _, g in w.groupby("grp") if len(g) >= 2]
    if len(groups) < 2:
        return dict(ok=False)
    stat, p = stats.levene(*groups)
    return dict(ok=True, stat=float(stat), p=float(p))


def pre_post_by_group(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp, g in w.groupby("grp"):
        n = len(g)
        row = dict(المجموعة=GROUPS.get(grp, grp), n=n,
                   قبلي_متوسط=g["pre"].mean(), قبلي_ع=g["pre"].std(),
                   بعدي_متوسط=g["post"].mean(), بعدي_ع=g["post"].std(),
                   فرق=g["gain"].mean())
        if n >= 3 and g["gain"].std() > 0:
            t, p = stats.ttest_rel(g["post"], g["pre"])
            row.update(t=float(t), p=float(p), d_z=float(g["gain"].mean() / g["gain"].std()))
        rows.append(row)
    return pd.DataFrame(rows)


def gains_anova(w: pd.DataFrame) -> dict:
    groups = {k: g["gain"].values for k, g in w.groupby("grp") if len(g) >= 2}
    if len(groups) < 2:
        return dict(ok=False)
    f, p = stats.f_oneway(*groups.values())
    out = dict(ok=True, F=float(f), p=float(p), groups=list(groups))
    if len(groups) >= 3:
        try:
            th = stats.tukey_hsd(*groups.values())
            names = list(groups)
            pairs = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    pairs.append((GROUPS.get(names[i], names[i]), GROUPS.get(names[j], names[j]),
                                  float(th.pvalue[i, j])))
            out["tukey"] = pairs
        except Exception:  # noqa: BLE001
            pass
    return out
