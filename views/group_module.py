"""وحدة المجموعة التجريبية: تحليل الفيديو + (تغذية راجعة بصرية | وصفة ذكية).

group = "cv"  -> التجريبية الأولى (الرؤية الحاسوبية): برنامج ثابت + تغذية راجعة بصرية بعد كل مجموعة.
group = "dl"  -> التجريبية الثانية (الشبكات العصبية العميقة): وصفة فردية تُحدَّث من بيانات اللاعب + نماذج تنبؤية.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from core import db, dnn, feedback, metrics, pose
from core import live as live_core, reference
from core.config import GROUPS, LEVELS, MOVE_TYPES, PHASES, ZONE_SCORES
from core.exercises import (DIFFICULTY_DEFAULT, allowed_moves, LIBRARY, N_UNITS, SKILL_LABELS, UNIT_PLAN,
                            default_prescription, get_exercise, plan_for_unit)
from core.synthetic import synthetic_session

from . import live as live_view, stimulus
from .common import ms


# ----------------------------------------------------------------- أدوات
def _cleanup_video() -> None:
    p = st.session_state.pop("vid_path", None)
    st.session_state.pop("vid_sig", None)
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass


def _store_upload(up) -> str:
    sig = f"{up.name}-{up.size}"
    if st.session_state.get("vid_sig") != sig:
        _cleanup_video()
        suffix = os.path.splitext(up.name)[1] or ".mp4"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(up.getbuffer())
        st.session_state["vid_path"] = path
        st.session_state["vid_sig"] = sig
        st.session_state.pop("analysis", None)
    return st.session_state["vid_path"]


def _ex_label(code: str) -> str:
    e = get_exercise(code)
    return f"{code} – {e.name}"


def _player_selector(group: str):
    players = db.list_players(group)
    with st.expander("➕ إضافة لاعب جديد"):
        c1, c2, c3 = st.columns([2, 2, 1])
        new_id = c1.text_input("رمز اللاعب (مثل P01 – لا تكتب الاسم)", key=f"{group}_newid")
        dom = c2.selectbox("اليد المسيطرة", ["right", "left"], format_func=lambda x: "يمنى" if x == "right" else "يسرى",
                           key=f"{group}_newdom")
        if c3.button("إضافة", key=f"{group}_addbtn"):
            if not new_id.strip():
                st.warning("اكتب رمز اللاعب.")
            elif db.add_player(new_id, group, dom):
                st.success("تمت الإضافة.")
                st.rerun()
            else:
                st.warning("هذا الرمز موجود مسبقاً.")
    if players.empty:
        st.info("لا يوجد لاعبون في هذه المجموعة بعد. أضف لاعباً من الأعلى.")
        return None, "right"
    pid = st.selectbox("اللاعب", players["player_id"].tolist(), key=f"{group}_pid")
    dom = players.loc[players["player_id"] == pid, "dominant"].iloc[0]
    return pid, dom


def _previous_summary(pid: str, level: int) -> dict | None:
    df = db.trials_df(pid)
    if df.empty:
        return None
    df = df[df["level"] == level]
    if df.empty:
        return None
    last_sid = df["session_id"].iloc[-1]
    return dnn.session_summary(df[df["session_id"] == last_sid])


@st.cache_data(show_spinner=False)
def _fit_cached(group: str, n_rows: int):
    df = db.trials_df()
    df = df[df["grp"] == group] if not df.empty else df
    model, msg = dnn.fit_accuracy_model(df)
    return model, msg


# ------------------------------------------------------------- تحليل أداء جديد
def _run_analysis(group, pid, dominant, ex, phase, unit, difficulty, method, params, fps_override,
                  crop, width, synthetic, move_type="free"):
    use_wrist = ex.use_wrist and move_type != "free"
    if synthetic:
        rng = np.random.default_rng(int.from_bytes(os.urandom(2), "little"))
        rts = list(np.clip(rng.normal(0.36, 0.07, ex.reps), 0.18, 0.8))
        kp, onsets = synthetic_session(120.0, rts, contact=ex.contact, seed=int(rng.integers(1000)))
        fps = 120.0
        xy = metrics.clean_keypoints(kp, fps)
        trials = metrics.analyze_trials(xy, fps, onsets, dominant, use_wrist, ex.contact)
        return dict(pid=pid, ex=ex.code, phase=phase, unit=unit, difficulty=difficulty, fps=fps,
                    trials=trials, keyframes=[], source="synthetic", detected=1.0, move_type=move_type)

    path = st.session_state.get("vid_path")
    prog = st.progress(0.0, text="جارٍ تقدير وضعية اللاعب في الفيديو…")
    res = pose.extract_keypoints(path, fps_override or None, width, crop,
                                 progress=lambda f: prog.progress(min(1.0, f), text="جارٍ تقدير وضعية اللاعب في الفيديو…"))
    prog.empty()
    fps = res.fps
    if method == "flash":
        series, _ = pose.read_roi_brightness(path, params["roi"], fps_override or None)
        onsets = metrics.detect_onsets_from_brightness(series, fps)
    elif method == "offsets":
        onsets = metrics.onsets_from_offsets(params["first"], params["offsets"])
    else:
        onsets = params["times"]
    if not onsets:
        raise ValueError("لم تُحدَّد أزمنة المثيرات. راجع طريقة تحديد المثير.")
    xy = metrics.clean_keypoints(res.kp, fps)
    trials = metrics.analyze_trials(xy, fps, onsets, dominant, use_wrist, ex.contact)

    keyframes = []
    for t in trials[:6]:
        idx = t["contact_frame"] if (ex.contact and t["contact_frame"] >= 0) else t["onset_frame"]
        if idx < 0:
            continue
        frame = pose.get_frame(path, idx)
        if frame is None:
            continue
        proc = pose.prep_frame(frame, crop, width)
        if proc.shape[1] != res.width or proc.shape[0] != res.height:
            continue
        labels = {}
        if ex.contact and np.isfinite(t.get("elbow", np.nan)):
            labels[metrics.LM["r_el" if dominant == "right" else "l_el"]] = f"{t['elbow']:.0f}"
        img = pose.draw_skeleton(proc, xy[idx], 1.0, labels)
        cap = f"المحاولة {t['idx']} – " + ("لحظة ضربة الشبكة" if ex.contact else "بداية الحركة")
        if ex.contact:
            stt = live_core.posture_status(live_core.frame_angles(xy[idx], dominant), True, True)
            ref = reference.draw_reference(reference.kind_for(True), 300, img.shape[0])
            img = reference.side_by_side(img, ref)
            cap += " (الأداء يسار، والوضع الصحيح يمين)"
        keyframes.append((cap, img[:, :, ::-1]))
    return dict(pid=pid, ex=ex.code, phase=phase, unit=unit, difficulty=difficulty, fps=fps,
                trials=trials, keyframes=keyframes, source="video", detected=res.detected_ratio, move_type=move_type)


def _show_results(group, a, dominant):
    ex = get_exercise(a["ex"])
    df = pd.DataFrame(a["trials"])
    df["acc_score"] = np.nan
    summ = dnn.session_summary(df)
    st.markdown("### نتائج التحليل")
    if a["detected"] < 0.6:
        st.warning(f"اكتُشف اللاعب في {a['detected']:.0%} فقط من الإطارات. تأكد أن اللاعب وحده في الإطار وأن الإضاءة جيدة.")
    c = st.columns(4)
    c[0].metric("متوسط زمن الاستجابة", ms(summ.get("mean_rt")))
    c[1].metric("أفضل محاولة", ms(summ.get("best_rt")))
    c[2].metric("محاولات صالحة", f"{summ.get('n_valid', 0)} / {summ.get('n', 0)}")
    c[3].metric("نسبة الاستباق", f"{summ.get('anticipation_rate', 0):.0%}")

    chart = df.set_index("idx")["rt"].rename("زمن الاستجابة (ث)")
    if chart.notna().any():
        st.bar_chart(chart)

    st.caption("راجع المحاولات: يمكنك استبعاد محاولة غير صحيحة، وإدخال منطقة سقوط الريشة (0 = خارج) لتكوين مؤشر الدقة.")
    view = pd.DataFrame({
        "اعتماد": df["valid"] == 1,
        "المحاولة": df["idx"],
        "زمن المثير (ث)": df["t_stim"].round(2),
        "زمن الاستجابة (م.ث)": (df["rt"] * 1000).round(0),
        "الحالة": df["flag"].map(feedback.FLAG_AR),
    })
    cfg = {"اعتماد": st.column_config.CheckboxColumn("اعتماد المحاولة")}
    if ex.contact:
        view["زاوية المرفق"] = df["elbow"].round(0)
        view["زاوية الركبة"] = df["knee"].round(0)
        view["ميل الجذع"] = df["trunk"].round(0)
        view["منطقة السقوط"] = pd.Series([None] * len(df), dtype="float")
        cfg["منطقة السقوط"] = st.column_config.NumberColumn(
            "منطقة السقوط (0 = خارج)", min_value=0, max_value=max(ZONE_SCORES), step=1)
    disabled = [c for c in view.columns if c not in ("اعتماد", "منطقة السقوط")]
    edited = st.data_editor(view, column_config=cfg, disabled=disabled, hide_index=True,
                            key=f"{group}_editor", use_container_width=True)

    if a["keyframes"]:
        with st.expander("🖼️ لقطات الهيكل الحركي (تغذية راجعة بصرية)", expanded=(group == "cv")):
            cols = st.columns(2)
            for i, (cap, img) in enumerate(a["keyframes"]):
                cols[i % 2].image(img, caption=cap, use_container_width=True)

    prev = _previous_summary(a["pid"], ex.level)
    st.markdown("#### التغذية الراجعة")
    for m in feedback.rt_feedback(summ, prev):
        st.write("• " + m)
    if ex.contact:
        for m in feedback.posture_feedback(a["trials"]):
            st.write("• " + m)
        st.caption("المدى المرجعي للوضع الحركي قيم مبدئية تعتمدها لجنة الخبراء قبل التطبيق.")
    if group == "dl" and ex.contact:
        model, _msg = _fit_cached("dl", len(db.trials_df()))
        ok = df[(df["valid"] == 1)].dropna(subset=dnn.FEATURES)
        if model is not None and len(ok):
            st.info(f"الدقة المتوقعة من الشبكة العصبية العميقة لهذه الجلسة: {float(np.mean(model.predict(ok))):.2f} درجة")
            recs = dnn.what_if(model, ok)
            if recs:
                st.markdown("**تحليل «ماذا لو» (الشبكة العميقة):**")
                for r in recs:
                    st.write(f"• تقريب «{r['label']}» من {r['current']:.1f} إلى نحو {r['target']:.1f} "
                             f"يرفع الدقة المتوقعة بمقدار {r['delta']:.2f} درجة.")
        elif model is None:
            st.caption("تُفعَّل توقعات الشبكة العميقة بعد توفر بيانات كافية من المجموعة: " + _msg)

    if st.button("💾 حفظ الجلسة في قاعدة البيانات", type="primary", key=f"{group}_save"):
        rows = []
        for i, t in enumerate(a["trials"]):
            r = dict(t)
            keep = bool(edited.loc[i, "اعتماد"])
            if keep != bool(t["valid"]):
                r["flag"] = "manual_include" if keep else "excluded"
            r["valid"] = int(keep and np.isfinite(t["rt"]))
            if ex.contact and "منطقة السقوط" in edited:
                z = edited.loc[i, "منطقة السقوط"]
                if pd.notna(z):
                    r["zone"] = int(z)
                    r["acc_score"] = float(ZONE_SCORES.get(int(z), 0))
            rows.append(r)
        sid = db.add_session(a["pid"], a["ex"], rows, unit=a["unit"], phase=a["phase"], fps=a["fps"],
                             source=a["source"], difficulty=a["difficulty"], move_type=a.get("move_type", "free"))
        st.session_state.pop("analysis", None)
        _cleanup_video()
        st.success(f"تم الحفظ (رقم الجلسة {sid}).")
        st.rerun()


def _selectors(group, pid):
    """اختيارات الجلسة المشتركة بين البث المباشر والتحليل بعد التمرين."""
    key = lambda s: f"{group}_{s}"
    c1, c2, c3, c4 = st.columns(4)
    phase = c1.selectbox("نوع الجلسة", list(PHASES), index=1, format_func=PHASES.get, key=key("phase"))
    players_df = db.trials_df(pid)
    last_unit = int(players_df["unit"].dropna().max()) if not players_df.empty and players_df["unit"].notna().any() else 0
    unit = None
    if phase == "train":
        unit = c2.selectbox("الوحدة", list(range(1, N_UNITS + 1)), index=min(last_unit, N_UNITS - 1) if last_unit else 0,
                            key=key("unit"))
        codes = list(plan_for_unit(unit)["codes"])
    else:
        c2.write("")
        codes = [e.code for e in LIBRARY]
    ex_code = c3.selectbox("التمرين", codes, format_func=_ex_label, key=key("ex"))
    ex = get_exercise(ex_code)
    moves = allowed_moves(ex)
    move_type = c4.selectbox("نوع الحركة", list(moves), format_func=MOVE_TYPES.get, key=key(f"move_{ex_code}"))

    difficulty = DIFFICULTY_DEFAULT
    if group == "dl":
        rec = DIFFICULTY_DEFAULT
        if unit:
            pres = dnn.prescribe(unit, db.trials_df(pid), _group_df("dl"), _last_difficulty(pid))
            rec = pres["difficulty"]
        difficulty = st.slider("مستوى الصعوبة (الموصى به من النظام مضبوط تلقائياً)", 1, 5, rec, key=key(f"diff_{unit}"))

    st.caption(f"المستوى: {LEVELS[ex.level]} • المثير: {ex.stimulus} • الاستجابة: {ex.response}")

    return phase, unit, ex, difficulty, move_type


def _analysis_tab(group, pid, dominant):
    """تبويب الأداء: خياران — بث مباشر أمام اللاعب، أو تحليل فيديو بعد التمرين."""
    key = lambda s: f"{group}_{s}"
    mode = st.radio("طريقة العمل", ["live", "video"], horizontal=True, key=key("mode"),
                    format_func={"live": "📡 بث مباشر (الأداء يُعرض أمام اللاعب لحظياً)",
                                 "video": "📹 تحليل بعد التمرين (رفع فيديو)"}.get)
    phase, unit, ex, difficulty, move_type = _selectors(group, pid)
    if mode == "live":
        live_view.render_live(group, pid, dominant, ex, phase, unit, difficulty, move_type)
        a = st.session_state.get("analysis")
        if a and a["pid"] == pid and a.get("source") == "live":
            _show_results(group, a, dominant)
        return
    _video_tab(group, pid, dominant, phase, unit, ex, difficulty, move_type)


def _video_tab(group, pid, dominant, phase, unit, ex, difficulty, move_type):
    key = lambda s: f"{group}_{s}"
    with st.expander("⚙️ إعدادات الفيديو والمثيرات", expanded=True):
        method = st.radio(
            "كيف نحدد لحظة ظهور المثير في الفيديو؟",
            ["offsets", "flash", "manual"], horizontal=True, key=key("method"),
            format_func={"offsets": "إزاحات مولّد المثيرات + زمن أول مثير",
                         "flash": "اكتشاف الوميض/تغير السطوع تلقائياً",
                         "manual": "قائمة أزمنة أدخلها يدوياً"}.get)
        up = st.file_uploader("فيديو الأداء (لاعب واحد داخل الإطار، كاميرا ثابتة)",
                              type=["mp4", "mov", "avi", "mkv", "m4v"], key=key("upl"))
        path = _store_upload(up) if up else None
        if not up:
            _cleanup_video()

        params: dict = {}
        if method == "offsets":
            a1, a2 = st.columns(2)
            first = a1.number_input("زمن أول مثير في الفيديو (ثانية)", 0.0, 600.0, 1.0, 0.01, key=key("first"))
            offs = a2.text_area("إزاحات المثيرات من المولّد (ثانية، مفصولة بفواصل)", key=key("offs"),
                                help="انسخها من حقل النتائج في تبويب مولّد المثيرات")
            params = dict(first=first, offsets=metrics.parse_float_list(offs) if offs.strip() else [0.0])
        elif method == "manual":
            tx = st.text_area("أزمنة المثيرات في الفيديو (ثانية، مفصولة بفواصل)", key=key("manual"))
            params = dict(times=metrics.parse_float_list(tx) if tx.strip() else [])
        else:
            b1, b2 = st.columns(2)
            xr = b1.slider("منطقة الشاشة في الفيديو – أفقياً", 0.0, 1.0, (0.0, 0.2), 0.01, key=key("rx"))
            yr = b2.slider("منطقة الشاشة في الفيديو – عمودياً", 0.0, 1.0, (0.0, 0.2), 0.01, key=key("ry"))
            params = dict(roi=(xr[0], yr[0], xr[1], yr[1]))
            if path:
                try:
                    import cv2
                    fr = pose.get_frame(path, 0)
                    if fr is not None:
                        h, w = fr.shape[:2]
                        cv2.rectangle(fr, (int(xr[0] * w), int(yr[0] * h)), (int(xr[1] * w), int(yr[1] * h)), (0, 0, 255), 4)
                        st.image(fr[:, :, ::-1], caption="المنطقة الحمراء = موضع الشاشة (يجب أن تظهر فيها الشاشة)",
                                 use_container_width=True)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"تعذر عرض المعاينة: {e}")

        a1, a2, a3 = st.columns(3)
        fps_override = a1.number_input("معدل الإطارات الحقيقي (0 = من الملف)", 0.0, 480.0, 0.0, 1.0, key=key("fps"),
                                       help="مقاطع الحركة البطيئة في الهواتف قد تحمل بيانات fps غير صحيحة؛ أدخل المعدل الفعلي (مثل 120 أو 240).")
        xr2 = a2.slider("حصر التحليل في منطقة اللاعب – أفقياً", 0.0, 1.0, (0.0, 1.0), 0.01, key=key("cx"))
        yr2 = a3.slider("حصر التحليل في منطقة اللاعب – عمودياً", 0.0, 1.0, (0.0, 1.0), 0.01, key=key("cy"))
        crop = None if (xr2 == (0.0, 1.0) and yr2 == (0.0, 1.0)) else (xr2[0], yr2[0], xr2[1], yr2[1])
        width = st.selectbox("دقة المعالجة (عرض الإطار)", [480, 640, 800], index=1, key=key("w"))

        status, msg = pose.backend_status()
        synthetic = st.checkbox("وضع التجربة: أداء اصطناعي بدل الفيديو (لتجربة الواجهة فقط – لا يُعدّ بيانات بحثية)",
                                key=key("synth"))
        if status == "none" and not synthetic:
            st.error(f"نموذج تقدير الوضعية غير متاح: {msg}")

    if st.button("▶️ حلّل الأداء", type="primary", key=key("run"),
                 disabled=(path is None and not synthetic)):
        try:
            st.session_state["analysis"] = _run_analysis(
                group, pid, dominant, ex, phase, unit, difficulty, method, params,
                fps_override, crop, width, synthetic, move_type)
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("analysis", None)
            st.error(f"تعذر التحليل: {e}")
    a = st.session_state.get("analysis")
    if a and a["pid"] == pid and a.get("source") != "live":
        _show_results(group, a, dominant)


# ------------------------------------------------------------ برنامج / وصفة
def _group_df(group):
    df = db.trials_df()
    return df[df["grp"] == group] if not df.empty else df


def _last_difficulty(pid):
    s = db.sessions_df()
    s = s[(s["player_id"] == pid) & (s["phase"] == "train") & s["difficulty"].notna()]
    return int(s["difficulty"].iloc[-1]) if len(s) else None


def _items_table(items):
    rows = []
    for it in items:
        e = get_exercise(it["code"])
        rows.append({"التمرين": f"{e.code} – {e.name}", "المستوى": LEVELS[e.level], "المثير": e.stimulus,
                     "المجموعات × التكرارات": f"{it['sets']} × {it['reps']}",
                     "الراحة بين التكرارات (ث)": it["rest_rep"], "الراحة بين المجموعات (ث)": it["rest_set"],
                     **({"سبب الاختيار": it["reason"]} if "reason" in it else {})})
    return pd.DataFrame(rows)


def _library_expander():
    """مكتبة التمرينات كاملة: أقسام الاستجابة الأربعة وأنواع الحركة المسموحة لكل تمرين."""
    with st.expander("📚 مكتبة تمرينات الاستجابة الحركية (الأقسام الأربعة)"):
        for lvl, name in LEVELS.items():
            st.markdown(f"**{lvl}. {name}**")
            rows = [{"الرمز": e.code, "التمرين": e.name, "المثير": e.stimulus, "الاستجابة المطلوبة": e.response,
                     "أنواع الحركة": "، ".join(MOVE_TYPES[m] for m in allowed_moves(e)),
                     "المجموعات × التكرارات": f"{e.sets} × {e.reps}"} for e in LIBRARY if e.level == lvl]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("الحركة بدون أداة والحركة مع المضرب متاحتان في المستويات 1–3، أما ضربات الشبكة (المستوى 4) "
                   "فتؤدى بمضرب، مع كرة الريشة أو بدونها.")


def _program_tab_cv():
    _library_expander()
    unit = st.selectbox("الوحدة", list(range(1, N_UNITS + 1)), key="cv_prog_unit")
    p = plan_for_unit(unit)
    st.markdown(f"**الأسبوع {p['week']} – الوحدة {unit}:** {p['goal']}")
    st.dataframe(_items_table(default_prescription(unit)), hide_index=True, use_container_width=True)
    st.caption("برنامج ثابت لجميع لاعبي المجموعة. الزمن الكلي للقسم الرئيسي 40 دقيقة.")


def _program_tab_dl(pid):
    _library_expander()
    df_p = db.trials_df(pid)
    last_unit = int(df_p["unit"].dropna().max()) if not df_p.empty and df_p["unit"].notna().any() else 0
    unit = st.selectbox("الوحدة", list(range(1, N_UNITS + 1)), index=min(last_unit, N_UNITS - 1) if last_unit else 0,
                        key="ml_prog_unit")
    gdf = _group_df("dl")
    pres = dnn.prescribe(unit, df_p, gdf, _last_difficulty(pid))
    p = plan_for_unit(unit)
    st.markdown(f"**الأسبوع {p['week']} – الوحدة {unit}:** {p['goal']}")
    c = st.columns(2)
    c[0].metric("الصعوبة الموصى بها (1–5)", pres["difficulty"])
    c[1].metric("أبرز نقطة ضعف", SKILL_LABELS.get(pres["weakness"], "لا توجد نقطة ضعف بارزة"))
    st.caption(pres["difficulty_reason"])
    st.dataframe(_items_table(pres["items"]), hide_index=True, use_container_width=True)
    if pres["profile"]:
        prof = pd.Series({SKILL_LABELS[k]: v for k, v in pres["profile"].items()}, name="درجة القصور (z)")
        st.markdown("#### ملف نقاط الضعف مقارنة بمتوسط المجموعة")
        st.bar_chart(prof)
        st.caption("قيمة موجبة أعلى = قصور أكبر عن متوسط المجموعة. تحتاج المقارنة إلى بيانات كافية من المجموعة.")
    series = dnn.per_session_series(df_p)
    f = dnn.forecast_rt(series)
    st.markdown("#### التنبؤ بمسار التحسن")
    if f["ok"]:
        st.write(f"ميل التحسن {f['slope'] * 1000:+.1f} م.ث لكل وحدة؛ المتوقع في الوحدة 12 ≈ {f['predicted'] * 1000:.0f} م.ث "
                 f"(R² = {f['r2']:.2f}).")
    else:
        st.caption(f["msg"])


def _models_tab():
    model, msg = _fit_cached("dl", len(db.trials_df()))
    st.markdown("#### الشبكة العصبية العميقة للتنبؤ بدقة ضربة الشبكة")
    st.caption(f"بنية الشبكة: {len(dnn.FEATURES)} مدخلاً ← طبقات مخفية {dnn.HIDDEN_LAYERS} (ReLU) ← مخرج واحد (درجة الدقة).")
    if model is None:
        st.info(msg)
        return
    c = st.columns(3)
    fmt = lambda v: "—" if v != v else f"{v:.2f}"
    c[0].metric("الشبكة العميقة (R²)", fmt(model.cv_r2))
    c[1].metric("الغابة العشوائية (R²)", fmt(model.baselines["rf"]))
    c[2].metric("الانحدار الخطي (R²)", fmt(model.baselines["linear"]))
    st.caption(f"عدد المحاولات: {model.n}. قيم R² بالتحقق المتقاطع حسب اللاعب (لاعبون غير مرئيين أثناء التدريب).")
    imp = pd.Series(model.importances, name="الأهمية")
    imp.index = [dnn.FEATURE_AR[i] for i in imp.index]
    st.bar_chart(imp)
    st.caption("أهمية المتغيرات (Permutation importance): أي المؤشرات الحركية يعتمد عليها النموذج أكثر في التنبؤ بالدقة.")


# ------------------------------------------------------------------ السجل
def _history_tab(pid):
    df = db.trials_df(pid)
    if df.empty:
        st.info("لا توجد جلسات محفوظة لهذا اللاعب.")
        return
    series = dnn.per_session_series(df)
    series["الجلسة"] = range(1, len(series) + 1)
    st.markdown("#### متوسط زمن الاستجابة عبر الجلسات (ثانية)")
    st.line_chart(series.set_index("الجلسة")["mean_rt"])
    show = df[["created", "phase", "unit", "exercise", "idx", "rt", "valid", "flag", "elbow", "knee", "trunk",
               "zone", "acc_score"]]
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.download_button("تنزيل بيانات اللاعب CSV", show.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{pid}_trials.csv", mime="text/csv")


# ------------------------------------------------------------------ الدخول
def render(group: str) -> None:
    st.subheader(GROUPS[group])
    pid, dominant = _player_selector(group)
    if pid is None:
        return
    names = ["📹 تحليل أداء جديد",
             "📅 برنامج الوحدة" if group == "cv" else "🧠 وصفة الوحدة الذكية",
             "🎯 مولّد المثيرات", "📈 سجل اللاعب"]
    if group == "dl":
        names.append("🤖 الشبكة العصبية العميقة")
    tabs = st.tabs(names)
    with tabs[0]:
        _analysis_tab(group, pid, dominant)
    with tabs[1]:
        if group == "cv":
            _program_tab_cv()
        else:
            _program_tab_dl(pid)
    with tabs[2]:
        codes = [e.code for e in LIBRARY]
        c1, c2 = st.columns(2)
        code = c1.selectbox("التمرين", codes, format_func=_ex_label, key=f"{group}_gen_ex")
        diff = c2.slider("الصعوبة", 1, 5, DIFFICULTY_DEFAULT, key=f"{group}_gen_diff") if group == "dl" else DIFFICULTY_DEFAULT
        stimulus.render_generator(get_exercise(code), diff)
        st.caption("ابدأ التصوير ثم اضغط «ابدأ». أول مثير = الزمن صفر في الإزاحات.")
    with tabs[3]:
        _history_tab(pid)
    if group == "dl":
        with tabs[4]:
            _models_tab()
