"""قاعدة بيانات SQLite: اللاعبون، الجلسات، المحاولات، نتائج الاختبارات.

لا تُخزَّن الفيديوهات ولا الأسماء: يُحفظ رمز اللاعب (مثل P01) والمؤشرات المستخرجة فقط،
حمايةً لخصوصية اللاعبين القاصرين.
"""
from __future__ import annotations

import io
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import DB_PATH, GROUPS

SCHEMA = """
CREATE TABLE IF NOT EXISTS players(
    player_id TEXT PRIMARY KEY,
    grp       TEXT NOT NULL CHECK(grp IN ('cv','dl','ctrl')),
    dominant  TEXT NOT NULL DEFAULT 'right',
    created   TEXT NOT NULL,
    is_demo   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id  TEXT NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
    unit       INTEGER,
    exercise   TEXT NOT NULL,
    phase      TEXT NOT NULL DEFAULT 'train',
    created    TEXT NOT NULL,
    fps        REAL,
    source     TEXT,
    difficulty INTEGER,
    notes      TEXT,
    move_type  TEXT NOT NULL DEFAULT 'free'
);
CREATE TABLE IF NOT EXISTS trials(
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    t_stim     REAL,
    rt         REAL,
    valid      INTEGER NOT NULL DEFAULT 0,
    flag       TEXT,
    elbow      REAL, shoulder REAL, knee REAL, trunk REAL, reach REAL, hit_height REAL, speed_peak REAL,
    zone       INTEGER,
    acc_score  REAL
);
CREATE TABLE IF NOT EXISTS test_results(
    player_id TEXT NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
    phase     TEXT NOT NULL CHECK(phase IN ('pre','post')),
    test      TEXT NOT NULL,
    value     REAL NOT NULL,
    PRIMARY KEY(player_id, phase, test)
);
"""

TRIAL_COLS = ["idx", "t_stim", "rt", "valid", "flag", "elbow", "shoulder", "knee",
              "trunk", "reach", "hit_height", "speed_peak", "zone", "acc_score"]

TEST_NAMES = {
    "reaction": "اختبار سرعة الاستجابة الحركية (ثانية)",
    "net_straight_acc": "دقة ضربة الشبكة المستقيمة (درجة)",
    "net_cross_acc": "دقة ضربة الشبكة القطرية (درجة)",
    "net_lift_acc": "دقة الرفع من الشبكة (درجة)",
}


@contextmanager
def connect(path: Optional[Path] = None):
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(path: Optional[Path] = None) -> None:
    with connect(path) as con:
        con.executescript(SCHEMA)
        cols = {r[1] for r in con.execute("PRAGMA table_info(sessions)")}
        if "move_type" not in cols:      # ترقية قواعد بيانات قديمة
            con.execute("ALTER TABLE sessions ADD COLUMN move_type TEXT NOT NULL DEFAULT 'free'")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _nan_to_none(v):
    if v is None:
        return None
    try:
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
    except TypeError:
        pass
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# -------------------------------------------------------------------- اللاعبون
def add_player(player_id: str, grp: str, dominant: str = "right", is_demo: int = 0,
               path: Optional[Path] = None) -> bool:
    player_id = player_id.strip().upper()
    if not player_id:
        raise ValueError("رمز اللاعب فارغ")
    if grp not in GROUPS:
        raise ValueError("مجموعة غير معروفة")
    with connect(path) as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO players(player_id,grp,dominant,created,is_demo) VALUES(?,?,?,?,?)",
            (player_id, grp, dominant, _now(), is_demo))
        return cur.rowcount > 0


def list_players(grp: Optional[str] = None, path: Optional[Path] = None) -> pd.DataFrame:
    q, args = "SELECT * FROM players", ()
    if grp:
        q, args = q + " WHERE grp=?", (grp,)
    with connect(path) as con:
        return pd.read_sql_query(q + " ORDER BY player_id", con, params=args)


def delete_player(player_id: str, path: Optional[Path] = None) -> None:
    with connect(path) as con:
        con.execute("DELETE FROM players WHERE player_id=?", (player_id,))


def clear_demo(path: Optional[Path] = None) -> int:
    with connect(path) as con:
        cur = con.execute("DELETE FROM players WHERE is_demo=1")
        return cur.rowcount


def delete_synthetic_sessions(path: Optional[Path] = None) -> int:
    with connect(path) as con:
        cur = con.execute("DELETE FROM sessions WHERE source='synthetic'")
        return cur.rowcount


# ---------------------------------------------------------------------- الجلسات
def add_session(player_id: str, exercise: str, trials: Iterable[dict], unit: Optional[int] = None,
                phase: str = "train", fps: Optional[float] = None, source: str = "video",
                difficulty: Optional[int] = None, notes: str = "", created: Optional[str] = None,
                path: Optional[Path] = None, move_type: str = "free") -> int:
    with connect(path) as con:
        cur = con.execute(
            "INSERT INTO sessions(player_id,unit,exercise,phase,created,fps,source,difficulty,notes,move_type)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (player_id, unit, exercise, phase, created or _now(), fps, source, difficulty, notes, move_type))
        sid = cur.lastrowid
        rows = []
        for t in trials:
            rows.append((sid,) + tuple(_nan_to_none(t.get(c)) for c in TRIAL_COLS))
        con.executemany(
            "INSERT INTO trials(session_id," + ",".join(TRIAL_COLS) + ") VALUES(?" + ",?" * len(TRIAL_COLS) + ")",
            rows)
        return sid


def sessions_df(path: Optional[Path] = None) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            "SELECT s.*, p.grp FROM sessions s JOIN players p USING(player_id) ORDER BY s.created, s.id", con)


def trials_df(player_id: Optional[str] = None, path: Optional[Path] = None) -> pd.DataFrame:
    q = ("SELECT t.*, s.player_id, s.unit, s.exercise, s.phase, s.created, s.difficulty, s.move_type, p.grp "
         "FROM trials t JOIN sessions s ON s.id=t.session_id JOIN players p ON p.player_id=s.player_id")
    args: tuple = ()
    if player_id:
        q += " WHERE s.player_id=?"
        args = (player_id,)
    q += " ORDER BY s.created, s.id, t.idx"
    with connect(path) as con:
        df = pd.read_sql_query(q, con, params=args)
    if not df.empty:
        df["level"] = df["exercise"].str[0].astype(int)
    else:
        df["level"] = pd.Series(dtype=int)
    return df


# ------------------------------------------------------------ نتائج الاختبارات
def set_test_result(player_id: str, phase: str, test: str, value: float, path: Optional[Path] = None) -> None:
    with connect(path) as con:
        con.execute("INSERT OR REPLACE INTO test_results(player_id,phase,test,value) VALUES(?,?,?,?)",
                    (player_id, phase, test, float(value)))


def tests_df(path: Optional[Path] = None) -> pd.DataFrame:
    with connect(path) as con:
        return pd.read_sql_query(
            "SELECT r.*, p.grp FROM test_results r JOIN players p USING(player_id) ORDER BY player_id", con)


# ----------------------------------------------------------- تصدير / نسخ احتياطي
def player_phase_summary(path: Optional[Path] = None) -> pd.DataFrame:
    df = trials_df(path=path)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for (pid, grp, phase), x in df.groupby(["player_id", "grp", "phase"]):
        v = x[(x["valid"] == 1) & x["rt"].notna()]
        rows.append(dict(
            player_id=pid, grp=grp, phase=phase, n_trials=len(x), n_valid=len(v),
            mean_rt=v["rt"].mean() if len(v) else np.nan,
            best_rt=v["rt"].min() if len(v) else np.nan,
            sd_rt=v["rt"].std() if len(v) > 1 else np.nan,
            anticipation_rate=float((x["flag"] == "anticipation").mean()),
            mean_accuracy=x["acc_score"].mean()))
    return pd.DataFrame(rows)


def export_excel_bytes(path: Optional[Path] = None) -> bytes:
    """ملف Excel جاهز للنقل إلى SPSS."""
    buf = io.BytesIO()
    tests = tests_df(path)
    wide = pd.DataFrame()
    if not tests.empty:
        wide = tests.pivot_table(index=["player_id", "grp"], columns=["test", "phase"], values="value")
        wide.columns = [f"{t}_{p}" for t, p in wide.columns]
        wide = wide.reset_index()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        list_players(path=path).to_excel(xw, sheet_name="players", index=False)
        sessions_df(path).to_excel(xw, sheet_name="sessions", index=False)
        trials_df(path=path).to_excel(xw, sheet_name="trials", index=False)
        player_phase_summary(path).to_excel(xw, sheet_name="player_summary", index=False)
        tests.to_excel(xw, sheet_name="tests_long", index=False)
        wide.to_excel(xw, sheet_name="tests_wide_spss", index=False)
    return buf.getvalue()


def backup_bytes(path: Optional[Path] = None) -> bytes:
    p = Path(path or DB_PATH)
    return p.read_bytes() if p.exists() else b""


def restore_from_bytes(data: bytes, path: Optional[Path] = None) -> None:
    p = Path(path or DB_PATH)
    tmp = p.with_suffix(".restore.tmp")
    tmp.write_bytes(data)
    try:
        con = sqlite3.connect(tmp)
        names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        con.close()
        if not {"players", "sessions", "trials", "test_results"} <= names:
            raise ValueError("الملف ليس نسخة احتياطية صالحة لهذا البرنامج")
        tmp.replace(p)
    finally:
        if tmp.exists():
            tmp.unlink()
