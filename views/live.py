"""شاشة البث المباشر: كاميرا اللاعب + الهيكل + المثير + الصورة المرجعية + تغذية راجعة لحظية.

تعتمد على streamlit-webrtc (الكاميرا تعمل داخل المتصفح). يُنصح بتشغيل التطبيق محلياً على حاسوب
القاعة المتصل بشاشة أمام اللاعب (أقل تأخيراً). على Streamlit Cloud قد تحتاج خادم TURN
(عيّن BAC_TURN_URL / BAC_TURN_USER / BAC_TURN_PASS في الأسرار).
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import streamlit as st

from core import live, pose, reference
from core.config import MOVE_TYPES

MOVE_NOTE = {
    "free": "بدون أداة: تنفيذ الحركة بالجسم فقط (اليد فارغة).",
    "racket": "مع مضرب: يمسك اللاعب المضرب وينفّذ الحركة دون ريشة (أداء ظلّي).",
    "racket_shuttle": "مع مضرب وكرة الريشة: يغذّي المدرب الريشة بعد الإشارة وينفّذ اللاعب الضربة فعلياً.",
}
STATUS_ICON = {"ok": "🟢", "low": "🟠 أقل من المدى", "high": "🔴 أعلى من المدى"}


def webrtc_available() -> tuple[bool, str]:
    try:
        import av  # noqa: F401
        import streamlit_webrtc  # noqa: F401
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def _rtc_config() -> dict:
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if os.getenv("BAC_TURN_URL"):
        servers.append({"urls": [os.environ["BAC_TURN_URL"]], "username": os.getenv("BAC_TURN_USER", ""),
                        "credential": os.getenv("BAC_TURN_PASS", "")})
    return {"iceServers": servers}


class LiveController:
    """يربط خيط الفيديو بالواجهة: يحمل الجلسة والصورة المرجعية بقفل."""

    def __init__(self):
        self.lock = threading.Lock()
        self.session: live.LiveSession | None = None
        self.pending: dict | None = None
        self.ref = None
        self.dominant = "right"
        self.analyze = None
        self.error = ""
        self.last_xy = None
        self.snap: dict = {}
        self.consumed = None

    # -- من الواجهة
    def request_start(self, **kw):
        with self.lock:
            self.pending = kw

    def request_stop(self):
        with self.lock:
            if self.session:
                self.session.stop()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.snap)

    def take_result(self):
        with self.lock:
            return self.session

    # -- من خيط الفيديو
    def process(self, bgr):
        t = time.perf_counter()
        if self.analyze is None:
            try:
                self.analyze, _ = pose.open_stream()
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                return bgr
        xy = self.analyze(bgr)
        with self.lock:
            if self.pending is not None:
                kw, self.pending = self.pending, None
                self.session = live.LiveSession(**kw["session"])
                self.ref = reference.draw_reference(kw["ref_kind"], 300, bgr.shape[0])
                self.dominant = kw["session"]["dominant"]
                self.session.start(t)
            state = self.session.update(t, xy) if self.session else dict(
                phase="idle", stim=None, status=live.posture_status(
                    live.frame_angles(xy, self.dominant), True, True) if xy is not None else {}, n_done=0, n_trials=0)
            if xy is not None:
                self.last_xy = xy
            self.snap = dict(state, message=(self.session.message if self.session else "الكاميرا جاهزة"))
            ref, dom = self.ref, self.dominant
        return live.render_frame(bgr, xy, state, dom, ref)


def _build_analysis(sess: "live.LiveSession", pid, ex_code, phase, unit, difficulty, move_type, ref_kind, dominant):
    rows = sess.result_rows()
    ref = reference.draw_reference(ref_kind, 300, 420)
    keyframes = []
    for r in sess.trials:
        if "_xy" in r:
            angles = live.frame_angles(r["_xy"], dominant)
            stt = live.posture_status(angles, True, True)
            keyframes.append((f"المحاولة {r['idx']} – أداؤك (يسار) مقابل الوضع الصحيح (يمين)",
                              live.pose_card(r["_xy"], ref, dominant, stt)[:, :, ::-1]))
    return dict(pid=pid, ex=ex_code, phase=phase, unit=unit, difficulty=difficulty, fps=None, trials=rows,
                keyframes=keyframes[:6], source="live", detected=sess.detected_ratio, move_type=move_type)


def render_live(group, pid, dominant, ex, phase, unit, difficulty, move_type):
    """يعرض شاشة البث ويعيد dict التحليل عند انتهاء الجلسة (أو None)."""
    ok, err = webrtc_available()
    if not ok:
        st.error("البث المباشر يحتاج المكتبتين streamlit-webrtc و av (موجودتان في requirements.txt). "
                 f"تفصيل الخطأ: {err}")
        return None
    status, msg = pose.backend_status()
    if status == "none":
        st.error(f"نموذج تقدير الوضعية غير متاح: {msg}")
        return None
    from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer
    import av

    st.info(MOVE_NOTE[move_type] + " — ضع الكاميرا أمام اللاعب (أو جانبه) بحيث يظهر جسمه كاملاً، "
            "وضع الشاشة بحيث يراها وهو يؤدي.")
    st.caption("زمن الاستجابة اللحظي يتضمن تأخير الشبكة والعرض (عشرات إلى مئات المللي ثانية)؛ "
               "هو للتدريب والتغذية الراجعة، والقياس البحثي الأدق من خيار «تحليل بعد التمرين» بفيديو عالي الإطارات.")
    n_trials = st.number_input("عدد التكرارات", 1, 30, int(ex.reps), key=f"{group}_live_n")

    ctrl_key = f"{group}_live_ctrl"
    if ctrl_key not in st.session_state:
        st.session_state[ctrl_key] = LiveController()
    ctrl: LiveController = st.session_state[ctrl_key]

    class Proc(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(ctrl.process(img), format="bgr24")

    ctx = webrtc_streamer(
        key=f"{group}_live_cam", mode=WebRtcMode.SENDRECV, rtc_configuration=_rtc_config(),
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": {"ideal": 30}}, "audio": False},
        video_processor_factory=Proc, async_processing=True)

    from core.exercises import apply_difficulty
    p = apply_difficulty(ex, difficulty)
    ref_kind = reference.kind_for(ex.contact)
    use_arm = ex.use_wrist and move_type != "free"
    cfg = dict(stim_cfg=dict(p["stim"]), n_trials=int(n_trials), dominant=dominant, use_wrist=use_arm,
               contact=bool(ex.contact), rest_s=max(1.5, p["rest_rep"] * 0.15 + 0.5))

    c1, c2, _ = st.columns([1, 1, 3])
    if ctx.state.playing:
        if c1.button("▶️ ابدأ الجلسة", type="primary", key=f"{group}_live_go"):
            ctrl.request_start(session=cfg, ref_kind=ref_kind)
        if c2.button("■ إيقاف", key=f"{group}_live_stop"):
            ctrl.request_stop()
    else:
        st.caption("اضغط START في الأعلى لتشغيل الكاميرا والسماح بالوصول إليها.")

    panel = st.empty()
    last = None
    while ctx.state.playing:
        s = ctrl.snapshot()
        with panel.container():
            if ctrl.error:
                st.error(ctrl.error)
            st.markdown(f"## {s.get('message', '')}")
            stim = s.get("stim")
            if stim is not None and s.get("phase") == "stimulus":
                st.markdown(f"### المثير: **{stim.label}**")
            st.progress(min(1.0, s.get("n_done", 0) / max(1, s.get("n_trials", 1) or 1)),
                        text=f"{s.get('n_done', 0)} من {s.get('n_trials', 0)}")
            for key, (val, stt, label) in (s.get("status") or {}).items():
                st.write(f"{STATUS_ICON[stt]} {label}: {val:.0f}")
        if s.get("phase") == "done":
            last = ctrl.take_result()
            break
        time.sleep(0.25)

    sess = last or (ctrl.take_result() if ctrl.session and ctrl.session.done else None)
    if sess is not None and sess.trials and ctrl.consumed is not sess:
        ctrl.consumed = sess
        a = _build_analysis(sess, pid, ex.code, phase, unit, difficulty, move_type, ref_kind, dominant)
        st.session_state["analysis"] = a
        return a
    return None
