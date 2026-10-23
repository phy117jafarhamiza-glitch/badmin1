"""عناصر واجهة مشتركة."""
from __future__ import annotations

import streamlit as st

from core.config import APP_SUBTITLE, APP_TITLE

RTL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Tajawal', sans-serif; }
.stApp, [data-testid="stSidebar"] { direction: rtl; }
.stMarkdown, .stText, label, p, h1, h2, h3, h4, li { text-align: right; }
.stTabs [data-baseweb="tab-list"] { direction: rtl; }
div[data-testid="stMetric"] { direction: rtl; text-align: right; }
div[data-testid="stAlert"] { direction: rtl; text-align: right; }
code, pre { direction: ltr; text-align: left; }
.small-note { color: #666; font-size: 0.85rem; }
"""


def inject_css() -> None:
    st.markdown(f"<style>{RTL_CSS}</style>", unsafe_allow_html=True)


def header(subtitle: str | None = None) -> None:
    st.title("🏸 " + APP_TITLE)
    st.caption(subtitle or APP_SUBTITLE)


def ms(x: float) -> str:
    return "—" if x is None or x != x else f"{x * 1000:.0f} م.ث"
