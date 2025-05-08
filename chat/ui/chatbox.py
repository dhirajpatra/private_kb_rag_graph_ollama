# ui/chatbox.py
import streamlit as st
import requests

API_URL = "http://rag_service:5000/chat"

def render_chatbox():
    st.markdown("### 💬 Assistant Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How can I help you with your case files today?"}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing case files..."):
                try:
                    response = requests.post(API_URL, json={"text": prompt})
                    response.raise_for_status()
                    reply = response.json().get("reply", "No reply received.")
                except Exception as e:
                    reply = f"Error: {e}"
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
