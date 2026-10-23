import unittest

import numpy as np

from core import metrics
from core.synthetic import brightness_series, synthetic_session


class TestMetrics(unittest.TestCase):
    def test_angle(self):
        self.assertAlmostEqual(metrics.angle_deg((0, 1), (0, 0), (1, 0)), 90, places=3)
        self.assertAlmostEqual(metrics.angle_deg((-1, 0), (0, 0), (1, 0)), 180, places=3)

    def test_reaction_time_recovered_at_several_fps(self):
        for fps in (60.0, 120.0, 240.0):
            true_rts = [0.22, 0.31, 0.40, 0.27]
            kp, onsets = synthetic_session(fps=fps, rts=true_rts, seed=3)
            xy = metrics.clean_keypoints(kp, fps)
            res = metrics.analyze_trials(xy, fps, onsets)
            self.assertEqual(len(res), 4)
            for r, t in zip(res, true_rts):
                self.assertEqual(r["flag"], "ok", r)
                # الحركة المصطنعة تتسارع تدريجياً؛ نقبل خطأ حتى 90 مللي ثانية
                self.assertLess(abs(r["rt"] - t), 0.09, (fps, r["rt"], t))

    def test_order_of_rt_preserved(self):
        kp, onsets = synthetic_session(fps=120, rts=[0.20, 0.45], seed=5)
        xy = metrics.clean_keypoints(kp, 120)
        res = metrics.analyze_trials(xy, 120, onsets)
        self.assertLess(res[0]["rt"], res[1]["rt"])

    def test_anticipation_flag(self):
        kp, onsets = synthetic_session(fps=120, rts=[0.02, 0.30], seed=2)
        xy = metrics.clean_keypoints(kp, 120)
        res = metrics.analyze_trials(xy, 120, onsets)
        self.assertIn(res[0]["flag"], ("anticipation", "unsteady"))
        self.assertEqual(res[1]["flag"], "ok")

    def test_no_response_when_stimulus_without_movement(self):
        kp, onsets = synthetic_session(fps=120, rts=[0.25, 0.25], gap=6.0, seed=1)
        xy = metrics.clean_keypoints(kp, 120)
        res = metrics.analyze_trials(xy, 120, [onsets[0], onsets[0] + 3.0])
        self.assertEqual(res[1]["flag"], "no_response")

    def test_contact_posture(self):
        kp, onsets = synthetic_session(fps=120, rts=[0.25, 0.30], contact=True, seed=4)
        xy = metrics.clean_keypoints(kp, 120)
        res = metrics.analyze_trials(xy, 120, onsets, use_wrist=True, contact=True)
        r = res[0]
        self.assertGreater(r["contact_frame"], r["onset_frame"])
        self.assertGreater(r["elbow"], 150)             # ذراع ممدودة تقريباً
        self.assertGreater(r["reach"], 0.8)             # المعصم ممدود أمام الورك
        self.assertTrue(np.isfinite(r["hit_height"]))

    def test_brightness_onsets(self):
        fps = 60
        onsets = [1.0, 3.5, 6.0]
        s = brightness_series(fps, onsets, 8.0)
        found = metrics.detect_onsets_from_brightness(s, fps)
        self.assertEqual(len(found), 3)
        for a, b in zip(found, onsets):
            self.assertAlmostEqual(a, b, delta=0.05)

    def test_offsets(self):
        self.assertEqual(metrics.onsets_from_offsets(2.0, [0.5, 3.0, 5.5]), [2.0, 4.5, 7.0])

    def test_parse(self):
        self.assertEqual(metrics.parse_float_list("1.5, 2،3\n4"), [1.5, 2.0, 3.0, 4.0])

    def test_clean_handles_missing_points(self):
        kp, onsets = synthetic_session(fps=60, rts=[0.3], seed=7)
        kp[100:110, 16, 3] = 0.0           # نقطة المعصم غير مرئية
        xy = metrics.clean_keypoints(kp, 60)
        self.assertTrue(np.isfinite(xy).all())


if __name__ == "__main__":
    unittest.main()
