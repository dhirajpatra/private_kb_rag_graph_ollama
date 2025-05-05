# chat/app.py

import streamlit as st
from ui.sidebar import render_sidebar
from ui.chatbox import render_chatbox
from ui.file_table import render_file_table

st.set_page_config(page_title="Zeno Chatbot", page_icon="📂", layout="wide")

render_sidebar()

st.markdown("### Initiate a workflow on these case files.")

render_file_table()

st.divider()

# ✅ Only this call is needed
render_chatbox()
