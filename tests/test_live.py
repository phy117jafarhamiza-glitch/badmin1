import unittest

import numpy as np

from core import config, db, live, reference
from core.live import LiveSession
from core.metrics import LM
from core.synthetic import _ease


def pose_xy(shift, arm):
    base = np.zeros((33, 2)); cx = 320.0
    base[LM["nose"]] = (cx, 180)
    base[LM["l_sh"]] = (cx - 25, 220); base[LM["r_sh"]] = (cx + 25, 220)
    base[LM["l_hip"]] = (cx - 15, 300); base[LM["r_hip"]] = (cx + 15, 300)
    base[LM["l_kn"]] = (cx - 15, 360); base[LM["r_kn"]] = (cx + 15, 360)
    base[LM["l_an"]] = (cx - 15, 420); base[LM["r_an"]] = (cx + 15, 420)
    base[LM["l_el"]] = (cx - 30, 265); base[LM["l_wr"]] = (cx - 30, 305)
    rest_el, rest_wr = np.array([cx + 30, 265.0]), np.array([cx + 30, 305.0])
    up_el, up_wr = np.array([cx + 62, 240.0]), np.array([cx + 100, 262.0])
    base[:, 0] += shift
    base[LM["r_el"]] = rest_el + (up_el - rest_el) * arm + [shift, 0]
    base[LM["r_wr"]] = rest_wr + (up_wr - rest_wr) * arm + [shift, 0]
    return base


def simulate(sess, rts, fps=30.0, contact=False, seconds=60.0):
    rng = np.random.default_rng(3)
    events, seen = [], set()
    t = 0.0
    sess.start(t)
    while t < seconds and not sess.done:
        if sess.phase == "stimulus" and sess._stim_t not in seen:
            seen.add(sess._stim_t)
            events.append(sess._stim_t + rts[len(events) % len(rts)])
        shift, arm = 0.0, 0.0
        for on in events:
            shift += 70 * _ease((t - on) / 0.45) - 70 * _ease((t - (on + 1.2)) / 0.45)
            if contact:
                arm += _ease((t - on) / 0.35) - _ease((t - (on + 0.55)) / 0.35)
        xy = pose_xy(shift, arm) + rng.normal(0, 0.3, (33, 2))
        sess.update(t, xy)
        t += 1.0 / fps
    return sess


class TestLive(unittest.TestCase):
    def test_rt_measured_close_to_truth(self):
        s = LiveSession(dict(mode="four_colors", delay_min=0.8, delay_max=1.2), n_trials=3, seed=1)
        simulate(s, [0.30, 0.40, 0.25])
        self.assertTrue(s.done)
        rows = s.result_rows()
        self.assertEqual(len(rows), 3)
        for r, truth in zip(rows, [0.30, 0.40, 0.25]):
            self.assertEqual(r["flag"], "ok")
            self.assertLess(abs(r["rt"] - truth), 0.15)    # حد الدقة عند 30 إطاراً/ث

    def test_no_response_and_anticipation(self):
        s = LiveSession(dict(mode="single_color", delay_min=0.8, delay_max=1.0), n_trials=1, seed=2)
        simulate(s, [50.0])             # لا استجابة قبل المهلة
        self.assertEqual(s.result_rows()[0]["flag"], "no_response")

    def test_contact_posture_recorded(self):
        s = LiveSession(dict(mode="skill_cue", skill="mixed", delay_min=0.8, delay_max=1.0), n_trials=2,
                        use_wrist=True, contact=True, seed=4)
        simulate(s, [0.3, 0.3], contact=True)
        r = s.result_rows()
        self.assertEqual(len(r), 2)
        self.assertTrue(np.isfinite(r[0]["reach"]) and np.isfinite(r[0]["elbow"]))
        self.assertGreater(r[0]["reach"], 0.8)

    def test_rows_save_to_db(self):
        import tempfile, pathlib
        s = LiveSession(dict(mode="arrows", delay_min=0.8, delay_max=1.0), n_trials=2, seed=5)
        simulate(s, [0.3, 0.3])
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.db"
            db.init_db(p); db.add_player("P1", "cv", path=p)
            sid = db.add_session("P1", "2-2", s.result_rows(), source="live", move_type="racket", path=p)
            self.assertGreater(sid, 0)
            self.assertEqual(db.sessions_df(path=p)["move_type"].iloc[0], "racket")

    def test_posture_status(self):
        st = live.posture_status(dict(elbow=90.0, shoulder=60.0, knee=120.0, trunk=20.0), True, True)
        self.assertEqual(st["elbow"][1], "low")
        self.assertEqual(st["knee"][1], "ok")
        st2 = live.posture_status(dict(elbow=90.0, knee=120.0, trunk=20.0), False, False)
        self.assertNotIn("elbow", st2)

    def test_reference_images(self):
        for k in ("ready", "net"):
            img = reference.draw_reference(k, 200, 300)
            self.assertEqual(img.shape, (300, 200, 3))
        both = reference.side_by_side(np.zeros((300, 400, 3), np.uint8), img)
        self.assertEqual(both.shape[0], 300)

    def test_move_types(self):
        from core.exercises import allowed_moves, get_exercise
        self.assertEqual(set(config.MOVE_TYPES), {"free", "racket", "racket_shuttle"})
        self.assertNotIn("free", allowed_moves(get_exercise("4-1")))
        self.assertIn("free", allowed_moves(get_exercise("1-1")))


if __name__ == "__main__":
    unittest.main()


class TestLiveController(unittest.TestCase):
    def test_controller_end_to_end(self):
        import sys, types
        from unittest import mock
        from tests import fake_streamlit
        fake_streamlit.install() if hasattr(fake_streamlit, "install") else None
        from views import live as lv
        ctrl = lv.LiveController()
        clock = {"t": 100.0}
        events = []

        def analyzer(img):
            t = clock["t"]
            sess = ctrl.session
            if sess is not None and sess.phase == "stimulus" and sess._stim_t not in [e[0] for e in events]:
                events.append((sess._stim_t, sess._stim_t + 0.3))
            shift, arm = 0.0, 0.0
            for _, on in events:
                shift += 70 * _ease((t - on) / 0.45) - 70 * _ease((t - (on + 1.2)) / 0.45)
                arm += _ease((t - on) / 0.35) - _ease((t - (on + 0.55)) / 0.35)
            return pose_xy(shift, arm)

        ctrl.analyze = analyzer
        img = np.full((480, 640, 3), 100, np.uint8)
        with mock.patch("views.live.time.perf_counter", lambda: clock["t"]):
            ctrl.request_start(session=dict(stim_cfg=dict(mode="skill_cue", skill="mixed", delay_min=0.8, delay_max=1.0),
                                            n_trials=2, dominant="right", use_wrist=True, contact=True),
                               ref_kind="net")
            out = None
            for _ in range(900):
                out = ctrl.process(img)
                clock["t"] += 1 / 30
                if ctrl.snapshot().get("phase") == "done":
                    break
        self.assertEqual(ctrl.snapshot()["phase"], "done")
        self.assertGreater(out.shape[1], 640)            # الصورة المرجعية ملصوقة
        a = lv._build_analysis(ctrl.session, "P1", "4-1", "pre", None, 3, "racket", "net", "right")
        self.assertEqual(len(a["trials"]), 2)
        self.assertEqual(a["move_type"], "racket")
        self.assertTrue(a["keyframes"])
