import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core import db, exercises, feedback, metrics, dnn, stats
from core.synthetic import seed_demo, synthetic_session


class TestExercises(unittest.TestCase):
    def test_library_and_plan(self):
        self.assertEqual(len(exercises.LIBRARY), 13)
        self.assertEqual(exercises.N_UNITS, 12)
        codes = {e.code for e in exercises.LIBRARY}
        for p in exercises.UNIT_PLAN:
            for c in p["codes"]:
                self.assertIn(c, codes)
        self.assertEqual([p["unit"] for p in exercises.UNIT_PLAN], list(range(1, 13)))
        # كل المستويات الأربعة ممثلة
        self.assertEqual({e.level for e in exercises.LIBRARY}, {1, 2, 3, 4})

    def test_difficulty_bounds(self):
        e = exercises.get_exercise("3-2")
        lo, hi = exercises.apply_difficulty(e, 1), exercises.apply_difficulty(e, 5)
        self.assertLess(lo["reps"], hi["reps"])
        self.assertGreater(lo["rest_set"], hi["rest_set"])
        self.assertLessEqual(hi["stim"]["fake_prob"], 0.7)
        self.assertGreaterEqual(hi["stim"]["delay_max"], hi["stim"]["delay_min"] + 0.5)


class TestDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.db"
        db.init_db(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_and_cascade(self):
        self.assertTrue(db.add_player("p01", "cv", path=self.path))
        self.assertFalse(db.add_player("P01", "cv", path=self.path))
        trials = [dict(idx=1, rt=0.3, valid=1, flag="ok", elbow=np.nan, zone=2, acc_score=2.0),
                  dict(idx=2, rt=0.05, valid=0, flag="anticipation")]
        sid = db.add_session("P01", "4-1", trials, unit=6, path=self.path)
        df = db.trials_df(path=self.path)
        self.assertEqual(len(df), 2)
        self.assertEqual(int(df["level"].iloc[0]), 4)
        self.assertTrue(pd.isna(df["elbow"].iloc[0]))
        db.set_test_result("P01", "pre", "reaction", 0.45, path=self.path)
        db.set_test_result("P01", "pre", "reaction", 0.44, path=self.path)   # استبدال
        self.assertEqual(len(db.tests_df(self.path)), 1)
        db.delete_player("P01", path=self.path)
        self.assertEqual(len(db.trials_df(path=self.path)), 0)
        self.assertEqual(len(db.tests_df(self.path)), 0)

    def test_invalid_group(self):
        with self.assertRaises(ValueError):
            db.add_player("X", "zzz", path=self.path)

    def test_demo_export_backup_restore(self):
        n = seed_demo(n_per_group=3, path=self.path)
        self.assertEqual(n, 9)
        xl = db.export_excel_bytes(self.path)
        sheets = pd.ExcelFile(__import__("io").BytesIO(xl)).sheet_names
        self.assertIn("tests_wide_spss", sheets)
        data = db.backup_bytes(self.path)
        self.assertGreater(len(data), 1000)
        other = Path(self.tmp.name) / "o.db"
        db.restore_from_bytes(data, other)
        self.assertEqual(len(db.list_players(path=other)), 9)
        with self.assertRaises(Exception):
            db.restore_from_bytes(b"not a database", other)
        self.assertEqual(db.clear_demo(self.path), 9)
        self.assertEqual(len(db.list_players(path=self.path)), 0)


class TestML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "m.db"
        db.init_db(cls.path)
        seed_demo(n_per_group=6, path=cls.path)
        cls.df = db.trials_df(path=cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_accuracy_model(self):
        model, msg = dnn.fit_accuracy_model(self.df)
        self.assertIsNotNone(model, msg)
        self.assertAlmostEqual(sum(model.importances.values()), 1.0, places=5)
        self.assertEqual(set(model.importances), set(dnn.FEATURES))
        pred = model.predict(self.df[self.df["acc_score"].notna()].head(5))
        self.assertEqual(len(pred), 5)
        self.assertIn("rf", model.baselines)
        self.assertIn("linear", model.baselines)
        valid = self.df[(self.df["valid"] == 1)].dropna(subset=dnn.FEATURES).head(30)
        recs = dnn.what_if(model, valid)
        self.assertLessEqual(len(recs), 3)
        for r in recs:
            self.assertIn(r["feature"], dnn.FEATURES)

    def test_model_refuses_small_data(self):
        model, msg = dnn.fit_accuracy_model(self.df.head(10))
        self.assertIsNone(model)
        self.assertIn("غير كافية", msg)

    def test_prescription(self):
        dl_df = self.df[self.df["grp"] == "dl"]
        pid = dl_df["player_id"].iloc[0]
        p_df = dl_df[(dl_df["player_id"] == pid) & (dl_df["unit"] <= 5)]
        pres = dnn.prescribe(6, p_df, dl_df, last_difficulty=3)
        codes = [i["code"] for i in pres["items"]]
        self.assertEqual(sorted(codes), sorted(exercises.plan_for_unit(6)["codes"]))
        self.assertTrue(1 <= pres["difficulty"] <= 5)
        for it in pres["items"]:
            self.assertIn("reason", it)

    def test_prescription_without_history(self):
        empty = self.df.iloc[0:0]
        pres = dnn.prescribe(1, empty, self.df, None)
        self.assertEqual(pres["difficulty"], exercises.DIFFICULTY_DEFAULT)
        self.assertEqual(len(pres["items"]), 3)

    def test_next_difficulty_rules(self):
        better = dict(mean_rt=0.38, anticipation_rate=0.0, miss_rate=0.0)
        worse = dict(mean_rt=0.46, anticipation_rate=0.0, miss_rate=0.0)
        base = dict(mean_rt=0.42, anticipation_rate=0.0, miss_rate=0.0)
        self.assertEqual(dnn.next_difficulty(3, better, base)[0], 4)
        self.assertEqual(dnn.next_difficulty(3, worse, base)[0], 2)
        self.assertEqual(dnn.next_difficulty(3, base, base)[0], 3)
        self.assertEqual(dnn.next_difficulty(5, better, base)[0], 5)
        self.assertEqual(dnn.next_difficulty(1, worse, base)[0], 1)

    def test_weakness_detects_slow_choice(self):
        grp = self.df[self.df["grp"] == "dl"]
        pid = grp["player_id"].iloc[0]
        p = grp[grp["player_id"] == pid].copy()
        p.loc[p["level"] == 2, "rt"] += 0.3
        prof = dnn.weakness_profile(p, grp)
        self.assertEqual(dnn.top_weakness(prof), "choice")

    def test_forecast(self):
        grp = self.df[self.df["grp"] == "cv"]
        pid = grp["player_id"].iloc[0]
        s = dnn.per_session_series(grp[grp["player_id"] == pid])
        f = dnn.forecast_rt(s)
        self.assertTrue(f["ok"])
        self.assertLess(f["slope"], 0)                 # التحسن = انخفاض زمن الاستجابة
        self.assertFalse(dnn.forecast_rt(s.head(2))["ok"])

    def test_stats(self):
        tests = db.tests_df(self.path)
        w = stats.wide_tests(tests, "reaction")
        self.assertEqual(len(w), 18)
        self.assertTrue(stats.homogeneity_levene(w)["ok"])
        tab = stats.pre_post_by_group(w)
        self.assertEqual(len(tab), 3)
        an = stats.gains_anova(w)
        self.assertTrue(an["ok"])
        self.assertIn("tukey", an)

    def test_summary_and_feedback(self):
        one = self.df[self.df["session_id"] == self.df["session_id"].iloc[0]]
        s = dnn.session_summary(one)
        msgs = feedback.rt_feedback(s, dict(s, mean_rt=s["mean_rt"] + 0.05))
        self.assertTrue(any("تحسّن" in m for m in msgs))
        rows = [dict(valid=1, elbow=160, shoulder=90, knee=120, trunk=10, reach=1.5, hit_height=0.5)]
        self.assertTrue(all("ضمن المدى" in m for m in feedback.posture_feedback(rows)))


if __name__ == "__main__":
    unittest.main()
