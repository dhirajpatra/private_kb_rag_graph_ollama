# ui/sidebar.py
import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.image("/app/chat/ui/images/00014.png", width=150)
        st.markdown("### Projects")
        st.selectbox("Select a case", ["Indian Young Lawyers’ Association v. State of Kerala"])
        st.markdown("### Menu")
        st.button("Assistant")
        st.button("Playbooks")
