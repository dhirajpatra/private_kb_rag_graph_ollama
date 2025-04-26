# chat/app.py

import streamlit as st
import requests

# API URL inside Docker
API_URL = "http://rag_service:5000/chat"

st.set_page_config(page_title="Simple Chat", page_icon="💬")
st.title("🧠 Private GPT Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi how can I help you today? you can ask me weather, market."}
    ]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Handle user input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={"text": prompt})
                response.raise_for_status()
                reply = response.json().get("reply", "No reply received.")
            except Exception as e:
                reply = f"Error: {e}"

            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

