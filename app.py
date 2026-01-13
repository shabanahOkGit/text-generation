import streamlit as st
from huggingface_hub import InferenceClient
import os

st.markdown(
    "<h1>🤖&nbsp;&nbsp;&nbsp;textGen search (with python ...)</h1>",
    unsafe_allow_html=True
)

HF_TOKEN = os.environ.get("HF_TOKEN") or st.secrets.get("HF_TOKEN")

if not HF_TOKEN:
    st.error("HF_TOKEN not found")
    st.stop()

client = InferenceClient(api_key=HF_TOKEN)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask something...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",
        messages=st.session_state.messages,
    )
    reply = response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
