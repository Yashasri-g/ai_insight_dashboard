import streamlit as st
from transformers import pipeline
import tempfile
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🎤 Voice AI Assistant", page_icon="🎧", layout="centered")

st.title("🎤 Voice-Powered AI Assistant")
st.write("Talk or type — your open-source AI listens, thinks, and replies!")

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    text_gen = pipeline("text-generation", model="distilgpt2")
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    return text_gen, whisper

text_gen, whisper = load_models()

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("🎙️ Speak or Type Your Message")

tabs = st.tabs(["🎧 Voice Input", "⌨️ Text Input"])
user_input = ""

with tabs[0]:
    uploaded_audio = st.file_uploader("Upload your voice (.wav, .mp3, or .flac)", type=["wav", "mp3", "flac"])
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_audio.getbuffer())
            tmp_path = tmp_file.name

        with st.spinner("Transcribing..."):
            transcription = whisper(tmp_path)["text"]

        os.remove(tmp_path)
        user_input = transcription
        st.success(f"🗣️ You said: {user_input}")

with tabs[1]:
    text_input = st.text_area("Type something:", placeholder="e.g. What is Artificial Intelligence?")
    if text_input:
        user_input = text_input

# -------------------------------
# GENERATE RESPONSE
# -------------------------------
if user_input:
    with st.spinner("Thinking..."):
        result = text_gen(user_input, max_length=80, do_sample=True, temperature=0.8)
        response = result[0]["generated_text"]

    st.markdown("### 🤖 AI Response:")
    st.write(response)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers")
