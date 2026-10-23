"""لوحة الباحث: إدارة اللاعبين، إدخال نتائج الاختبارات، مقارنة المجموعات، التصدير والنسخ الاحتياطي."""
from __future__ import annotations

import platform

import pandas as pd
import streamlit as st

from core import config, db, pose, stats
from core.config import GROUPS, PHASES
from core.db import TEST_NAMES
from core.synthetic import seed_demo


def render() -> None:
    st.subheader("لوحة الباحث")
    tabs = st.tabs(["👥 اللاعبون", "📝 نتائج الاختبارات", "📊 مقارنة المجموعات", "🏋️ بيانات التدريب",
                    "💾 التصدير والنسخ الاحتياطي", "🔧 حالة النظام"])
    with tabs[0]:
        _players()
    with tabs[1]:
        _tests_entry()
    with tabs[2]:
        _compare()
    with tabs[3]:
        _training()
    with tabs[4]:
        _export()
    with tabs[5]:
        _system()


def _players() -> None:
    c1, c2, c3, c4 = st.columns([2, 3, 2, 1])
    pid = c1.text_input("رمز اللاعب", key="r_pid")
    grp = c2.selectbox("المجموعة", list(GROUPS), format_func=GROUPS.get, key="r_grp")
    dom = c3.selectbox("اليد المسيطرة", ["right", "left"], format_func=lambda x: "يمنى" if x == "right" else "يسرى",
                       key="r_dom")
    if c4.button("إضافة", key="r_add"):
        if not pid.strip():
            st.warning("اكتب رمز اللاعب.")
        elif db.add_player(pid, grp, dom):
            st.success("تمت الإضافة.")
            st.rerun()
        else:
            st.warning("الرمز موجود مسبقاً.")
    df = db.list_players()
    if df.empty:
        st.info("لا يوجد لاعبون بعد.")
        return
    show = df.copy()
    show["grp"] = show["grp"].map(GROUPS)
    show["is_demo"] = show["is_demo"].map({0: "", 1: "تجريبي"})
    show.columns = ["الرمز", "المجموعة", "اليد", "تاريخ الإضافة", "ملاحظة"]
    st.dataframe(show, hide_index=True, use_container_width=True)
    counts = df.groupby("grp").size().rename(index=GROUPS)
    st.caption("أعداد المجموعات: " + "، ".join(f"{k}: {v}" for k, v in counts.items()))
    with st.expander("🗑️ حذف لاعب (يحذف جلساته ونتائجه)"):
        victim = st.selectbox("اللاعب", df["player_id"].tolist(), key="r_del")
        sure = st.checkbox("أؤكد الحذف النهائي", key="r_del_ok")
        if st.button("حذف", disabled=not sure, key="r_del_btn"):
            db.delete_player(victim)
            st.rerun()


def _tests_entry() -> None:
    st.caption("نتائج الاختبارات القبلية والبعدية المعيارية (بأدوات القياس المعتمدة في الفصل الثالث) – مستقلة عن تحليل الذكاء الاصطناعي.")
    players = db.list_players()
    if players.empty:
        st.info("أضف لاعبين أولاً.")
        return
    c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
    pid = c1.selectbox("اللاعب", players["player_id"].tolist(), key="t_pid")
    phase = c2.selectbox("الاختبار", ["pre", "post"], format_func=PHASES.get, key="t_phase")
    test = c3.selectbox("المتغير", list(TEST_NAMES), format_func=TEST_NAMES.get, key="t_test")
    val = c4.number_input("القيمة", value=0.0, step=0.01, format="%.3f", key="t_val")
    if st.button("حفظ النتيجة", key="t_save"):
        db.set_test_result(pid, phase, test, val)
        st.success("تم الحفظ (تُستبدل القيمة السابقة لنفس اللاعب والاختبار).")
    up = st.file_uploader("أو رفع ملف CSV بالأعمدة: player_id, phase, test, value", type=["csv"], key="t_csv")
    if up is not None and st.button("استيراد الملف", key="t_import"):
        try:
            d = pd.read_csv(up)
            ok, bad = 0, 0
            valid_ids = set(players["player_id"])
            for r in d.itertuples(index=False):
                pid_, ph, ts, v = str(r.player_id).strip().upper(), str(r.phase).strip(), str(r.test).strip(), float(r.value)
                if pid_ in valid_ids and ph in ("pre", "post") and ts in TEST_NAMES:
                    db.set_test_result(pid_, ph, ts, v)
                    ok += 1
                else:
                    bad += 1
            st.success(f"تم استيراد {ok} قيمة" + (f"، وتجاهل {bad} صفاً غير صالح." if bad else "."))
        except Exception as e:  # noqa: BLE001
            st.error(f"تعذر قراءة الملف: {e}")
    t = db.tests_df()
    if not t.empty:
        w = t.pivot_table(index=["player_id", "grp"], columns=["test", "phase"], values="value")
        w.columns = [f"{TEST_NAMES.get(a, a)} – {PHASES.get(b, b)}" for a, b in w.columns]
        st.dataframe(w.reset_index(), hide_index=True, use_container_width=True)


def _compare() -> None:
    st.caption("فحوص أولية سريعة. التحليل الإحصائي الرسمي للأطروحة يُجرى في SPSS من ملف التصدير.")
    tests = db.tests_df()
    if tests.empty:
        st.info("لا توجد نتائج اختبارات بعد.")
        return
    test = st.selectbox("المتغير", list(TEST_NAMES), format_func=TEST_NAMES.get, key="c_test")
    w = stats.wide_tests(tests, test)
    if w.empty:
        st.info("لا توجد بيانات قبلية وبعدية معاً لهذا المتغير.")
        return
    lev = stats.homogeneity_levene(w)
    st.markdown("**تجانس المجموعات في الاختبار القبلي (Levene)**")
    if lev["ok"]:
        st.write(f"F = {lev['stat']:.3f}، مستوى الدلالة p = {lev['p']:.3f} → " +
                 ("المجموعات متجانسة (p > 0.05)." if lev["p"] > 0.05 else "المجموعات غير متجانسة."))
    else:
        st.caption("يلزم لاعبان على الأقل في مجموعتين.")
    tab = stats.pre_post_by_group(w)
    st.markdown("**المتوسطات والانحرافات قبل/بعد (واختبار t المترابط داخل كل مجموعة)**")
    st.dataframe(tab.round(4), hide_index=True, use_container_width=True)
    gain = w.groupby("grp")["gain"].mean().rename(index=GROUPS)
    st.bar_chart(gain.rename("متوسط الفرق (بعدي − قبلي)"))
    an = stats.gains_anova(w)
    st.markdown("**مقارنة الفروق بين المجموعات (تحليل التباين الأحادي)**")
    if an["ok"]:
        st.write(f"F = {an['F']:.3f}، p = {an['p']:.4f}")
        if "tukey" in an:
            st.dataframe(pd.DataFrame(an["tukey"], columns=["المجموعة الأولى", "المجموعة الثانية", "p (Tukey)"]).round(4),
                         hide_index=True, use_container_width=True)
    else:
        st.caption("يلزم لاعبان على الأقل في مجموعتين.")
    if test == "reaction":
        st.caption("في زمن الاستجابة الانخفاض = تحسّن؛ وفي اختبارات الدقة الارتفاع = تحسّن.")


def _training() -> None:
    df = db.trials_df()
    df = df[df["phase"] == "train"] if not df.empty else df
    if df.empty:
        st.info("لا توجد جلسات تدريب محفوظة بعد.")
        return
    ser = []
    for (grp, unit), g in df.groupby(["grp", "unit"]):
        v = g[g["valid"] == 1]
        ser.append(dict(المجموعة=GROUPS.get(grp, grp), الوحدة=unit, mean_rt=v["rt"].mean()))
    pivot = pd.DataFrame(ser).pivot(index="الوحدة", columns="المجموعة", values="mean_rt")
    st.markdown("**متوسط زمن الاستجابة (ثانية) حسب الوحدة والمجموعة**")
    st.line_chart(pivot)
    s = db.sessions_df()
    s["grp"] = s["grp"].map(GROUPS)
    st.markdown("**الجلسات المحفوظة**")
    st.dataframe(s, hide_index=True, use_container_width=True)


def _export() -> None:
    st.markdown("#### التصدير")
    st.download_button("⬇️ ملف Excel كامل (جاهز لـ SPSS)", db.export_excel_bytes(),
                       file_name="badminton_ai_study.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("#### النسخ الاحتياطي")
    st.warning("على استضافة Streamlit Community Cloud يُمسح التخزين المحلي عند إعادة التشغيل. "
               "نزّل نسخة احتياطية بعد كل جلسة تدريبية، وأعد رفعها عند الحاجة.")
    st.download_button("⬇️ نسخة احتياطية من قاعدة البيانات (.db)", db.backup_bytes(), file_name="study_backup.db",
                       mime="application/octet-stream")
    up = st.file_uploader("استعادة نسخة احتياطية (.db) – تستبدل البيانات الحالية", type=["db"], key="restore_up")
    sure = st.checkbox("أؤكد استبدال البيانات الحالية", key="restore_ok")
    if up is not None and st.button("استعادة", disabled=not sure, key="restore_btn"):
        try:
            db.restore_from_bytes(up.getvalue())
            st.success("تمت الاستعادة.")
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"فشلت الاستعادة: {e}")

    st.markdown("#### بيانات تجريبية")
    st.caption("لاعبون وهميون لتجربة اللوحات والنماذج فقط. احذفها قبل بدء التجربة الحقيقية.")
    c1, c2, c3 = st.columns(3)
    if c1.button("إنشاء بيانات تجريبية", key="demo_seed"):
        n = seed_demo(6)
        st.success(f"أُضيف {n} لاعباً تجريبياً.")
        st.rerun()
    if c2.button("حذف اللاعبين التجريبيين", key="demo_clear"):
        st.success(f"حُذف {db.clear_demo()} لاعباً.")
        st.rerun()
    if c3.button("حذف جلسات «وضع التجربة»", key="synth_clear"):
        st.success(f"حُذفت {db.delete_synthetic_sessions()} جلسة.")
        st.rerun()


def _system() -> None:
    status, msg = pose.backend_status()
    st.write(f"نموذج تقدير الوضعية: **{status}** – {msg}")
    st.write(f"Python {platform.python_version()}")
    import numpy, sklearn  # noqa: E401
    st.write(f"numpy {numpy.__version__} • pandas {pd.__version__} • scikit-learn {sklearn.__version__} • streamlit {st.__version__}")
    st.write(f"قاعدة البيانات: `{config.DB_PATH}`")
    if config.using_default_codes():
        st.error("رموز الدخول الافتراضية ما زالت مستخدمة! عيّن BAC_CODE_RESEARCHER و BAC_CODE_CV و BAC_CODE_DL "
                 "في أسرار Streamlit قبل إتاحة التطبيق للاعبين.")
    else:
        st.success("رموز الدخول مخصّصة.")
