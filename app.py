import streamlit as st
import speech_recognition as sr
from transformers import pipeline

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🎤 Voice AI Assistant", page_icon="🎧", layout="centered")

st.title("🎤 Voice-Powered AI Assistant")
st.write("Talk or type — your open-source AI listens, thinks, and replies!")

# -------------------------------
# LOAD MODEL (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

ai = load_model()

# -------------------------------
# SPEECH TO TEXT FUNCTION
# -------------------------------
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn’t understand that."
    except Exception as e:
        return f"Error: {e}"

# -------------------------------
# INPUT SECTION
# -------------------------------
st.subheader("🎙️ Speak or Type Your Message")

tabs = st.tabs(["🎧 Voice Input", "⌨️ Text Input"])
user_input = ""

with tabs[0]:
    uploaded_audio = st.file_uploader("Upload your voice (.wav or .flac)", type=["wav", "flac"])
    if uploaded_audio:
        with open("temp.wav", "wb") as f:
            f.write(uploaded_audio.getbuffer())
        with st.spinner("Transcribing..."):
            user_input = transcribe_audio("temp.wav")
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
        result = ai(user_input, max_length=80, do_sample=True, temperature=0.8)
        response = result[0]["generated_text"]

    st.markdown("### 🤖 AI Response:")
    st.write(response)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers")
