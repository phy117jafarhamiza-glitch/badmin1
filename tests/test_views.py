"""اختبار دخان لصفحات Streamlit عبر محاكٍ (بدون تثبيت streamlit)."""
import os
import sys
import tempfile
import unittest


from tests import fake_streamlit  # noqa: E402

ST = fake_streamlit.install()

from core import config, db  # noqa: E402
from core.synthetic import seed_demo  # noqa: E402

db.init_db()
seed_demo(4)
db.add_player("LIVE01", "cv")
db.add_player("LIVE02", "dl")

from views import group_module, researcher, stimulus  # noqa: E402
from core.exercises import LIBRARY  # noqa: E402


def run(fn, *a, overrides=None, state=None):
    ST.calls.clear()
    ST.overrides = overrides or {}
    if state is not None:
        ST.session_state = state
    try:
        fn(*a)
    except fake_streamlit.RerunException:
        return "rerun"
    return "done"


class TestViews(unittest.TestCase):
    def test_cv_and_ml_render_empty_state(self):
        for g in ("cv", "dl"):
            self.assertEqual(run(group_module.render, g, state={}), "done")

    def test_demo_player_pages(self):
        for g, pid in (("cv", "D-CV01"), ("dl", "D-DL01")):
            st = {f"{g}_pid": pid}
            self.assertEqual(run(group_module.render, g, state=st), "done")

    def test_synthetic_analysis_and_save_all_exercises(self):
        before = len(db.sessions_df())
        n = 0
        for g, pid in (("cv", "LIVE01"), ("dl", "LIVE02")):
            for ex in LIBRARY:
                st = {f"{g}_pid": pid}
                ov = {f"{g}_mode": "video", f"{g}_phase": "pre", f"{g}_ex": ex.code, f"{g}_synth": True, f"{g}_run": True,
                      f"{g}_save": True}
                self.assertEqual(run(group_module.render, g, overrides=ov, state=st), "rerun", (g, ex.code))
                n += 1
        self.assertEqual(len(db.sessions_df()) - before, n)
        self.assertTrue(set(db.sessions_df()["move_type"]) <= set(config.MOVE_TYPES))
        t = db.trials_df("LIVE01")
        self.assertTrue((t["source"] if "source" in t else t["flag"]).notna().all())
        # لاعب ML بعد وجود جلسات: وصفة وبرنامج ونماذج
        st = {"ml_pid": "LIVE02"}
        self.assertEqual(run(group_module.render, "dl", state=st), "done")

    def test_live_mode_renders_without_camera_lib(self):
        # بدون streamlit-webrtc يجب أن تظهر رسالة واضحة لا استثناء
        st = {"cv_pid": "LIVE01"}
        self.assertEqual(run(group_module.render, "cv", overrides={"cv_mode": "live"}, state=st), "done")

    def test_move_type_selection_saved(self):
        st = {"cv_pid": "LIVE01"}
        ov = {"cv_mode": "video", "cv_phase": "pre", "cv_ex": "1-3", "cv_move_1-3": "racket",
              "cv_synth": True, "cv_run": True, "cv_save": True}
        run(group_module.render, "cv", overrides=ov, state=st)
        s = db.sessions_df()
        self.assertEqual(s.iloc[-1]["move_type"], "racket")

    def test_train_phase_flow(self):
        st = {"cv_pid": "LIVE01"}
        ov = {"cv_mode": "video", "cv_phase": "train", "cv_unit": 6, "cv_ex": "4-1", "cv_synth": True, "cv_run": True}
        self.assertEqual(run(group_module.render, "cv", overrides=ov, state=st), "done")
        self.assertIn("analysis", st)
        trials = st["analysis"]["trials"]
        self.assertEqual(len(trials), 6)
        self.assertTrue(any(t["valid"] for t in trials))

    def test_researcher_all_tabs(self):
        self.assertEqual(run(researcher.render, state={}), "done")

    def test_researcher_actions(self):
        ov = {"r_pid": "XQ1", "r_add": True}
        self.assertEqual(run(researcher.render, overrides=ov, state={}), "rerun")
        self.assertIn("XQ1", db.list_players()["player_id"].tolist())
        ov = {"t_pid": "D-CV01", "t_val": 0.41, "t_save": True}
        run(researcher.render, overrides=ov, state={})
        self.assertAlmostEqual(
            db.tests_df().query("player_id=='D-CV01' and phase=='pre' and test=='reaction'")["value"].iloc[0], 0.41)

    def test_generator_html_is_valid(self):
        for ex in LIBRARY:
            ST.calls.clear()
            stimulus.render_generator(ex, 3)
            html = ST.calls[-1][1][0]
            self.assertNotIn("__CFG__", html)
            self.assertIn(f'"mode": "{ex.stim["mode"]}"', html)


if __name__ == "__main__":
    unittest.main()
