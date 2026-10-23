"""نقطة الدخول: streamlit run app.py"""
import os

import streamlit as st

st.set_page_config(page_title="مدرّب الريشة الطائرة الذكي", page_icon="🏸", layout="wide")

# نقل الأسرار (إن وُجدت) إلى متغيرات البيئة قبل استيراد الإعدادات
try:
    for _k in ("BAC_CODE_RESEARCHER", "BAC_CODE_CV", "BAC_CODE_DL", "BAC_DEVELOPER", "BAC_TURN_URL", "BAC_TURN_USER", "BAC_TURN_PASS"):
        if _k in st.secrets:
            os.environ.setdefault(_k, str(st.secrets[_k]))
except Exception:  # noqa: BLE001  (لا يوجد ملف أسرار)
    pass

from core import config, db  # noqa: E402
from views import group_module, researcher  # noqa: E402
from views.common import header, inject_css  # noqa: E402

inject_css()
db.init_db()

ROLE_LABEL = {
    "researcher": "الباحث",
    "cv": "التجريبية الأولى – الرؤية الحاسوبية",
    "dl": "التجريبية الثانية – الشبكات العصبية العميقة",
}


def login() -> None:
    header()
    st.markdown("### تسجيل الدخول")
    code = st.text_input("رمز الدخول", type="password", key="login_code")
    if st.button("دخول", type="primary", key="login_btn"):
        codes = config.get_access_codes()
        role = next((r for r, c in codes.items() if code and code == c), None)
        if role:
            st.session_state["role"] = role
            st.rerun()
        else:
            st.error("رمز الدخول غير صحيح.")
    if config.using_default_codes():
        st.warning("تنبيه للمطوّر: رموز الدخول الافتراضية مفعّلة. غيّرها من إعدادات الأسرار قبل النشر.")
    st.caption("يستعمل المدرب رمز مجموعته. لا تُحفظ الفيديوهات ولا الأسماء، بل رموز اللاعبين والمؤشرات المستخرجة فقط.")


def main() -> None:
    role = st.session_state.get("role")
    if role not in ROLE_LABEL:
        login()
        return
    with st.sidebar:
        st.markdown(f"**الدور:** {ROLE_LABEL[role]}")
        view = role
        if role == "researcher":
            view = st.radio("القسم", ["researcher", "cv", "dl"], key="nav",
                            format_func={"researcher": "لوحة الباحث", "cv": "وحدة الرؤية الحاسوبية",
                                         "dl": "وحدة الشبكات العصبية العميقة"}.get)
        if st.button("تسجيل الخروج", key="logout"):
            st.session_state.clear()
            st.rerun()
        st.divider()
        st.caption(os.getenv("BAC_DEVELOPER", config.DEVELOPER))
    header()
    if view == "researcher":
        researcher.render()
    else:
        group_module.render(view)


main()
