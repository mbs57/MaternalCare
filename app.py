# app.py
import streamlit as st
from utils import apply_global_css
from home_page import render_home
from general_model_page import render_general_model

st.set_page_config(
    page_title="MaternalCare",
    page_icon="🤱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_global_css()

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

page = st.session_state["page"]

if page == "Home":
    render_home()
elif page == "General":
    render_general_model()
else:
    render_home()