import streamlit as st
from streamlit_audio_recorder import st_audio_recorder
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import numpy as np
import io
import soundfile as sf
from gtts import gTTS
import os

# ───────────────────────────────────────────────
# SETUP
st.set_page_config(page_title="Voice Biometric Assistant", page_icon="🎙️", layout="wide")
st.sidebar.title("🎧 Voice Biometric Assistant")
st.sidebar.write("""
- 🎙️ Record your voice in real time  
- 🧠 Identify who you are  
- 💬 Get a text + audio response
""")

encoder = VoiceEncoder()
registered_speakers = {}

# ───────────────────────────────────────────────
# SECTION: Speaker Registration
st.header("🔐 Register Speaker")

st.write("Press record, say a few words (5–10 seconds), and enter your name to register.")

reg_audio = st_audio_recorder(key="reg")

if reg_audio:
    st.audio(reg_audio, format="audio/wav")
    name = st.text_input("Enter your name:", key="name")
    if st.button("Register Voice"):
        wav_data, sr = sf.read(io.BytesIO(reg_audio))
        wav = preprocess_wav(wav_data)
        embedding = encoder.embed_utterance(wav)
        registered_speakers[name] = embedding
        st.success(f"✅ Speaker '{name}' registered successfully!")

# ───────────────────────────────────────────────
# SECTION: Speaker Verification
st.header("🎤 Identify Speaker")

st.write("Record your voice again — we'll try to identify who you are.")

test_audio = st_audio_recorder(key="test")

if test_audio:
    st.audio(test_audio, format="audio/wav")

    wav_data, sr = sf.read(io.BytesIO(test_audio))
    wav = preprocess_wav(wav_data)
    test_embedding = encoder.embed_utterance(wav)

    if registered_speakers:
        similarities = {name: 1 - cosine(test_embedding, emb)
                        for name, emb in registered_speakers.items()}
        identified_name = max(similarities, key=similarities.get)
        confidence = similarities[identified_name]

        st.markdown(f"### 🧭 Identified as: **{identified_name}** (confidence: {confidence:.2f})")

        if confidence > 0.7:
            response_text = f"Hello {identified_name}, welcome back! How can I assist you today?"
        else:
            response_text = "Sorry, I couldn’t confidently identify you. Please try again."

        st.write("💬 Response:", response_text)

        # Convert response to speech
        tts = gTTS(response_text)
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3")

        # Sidebar summary
        st.sidebar.markdown("### 🔎 Detection Summary")
        st.sidebar.write(f"**Speaker:** {identified_name}")
        st.sidebar.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("⚠️ No registered speakers found. Please register first.")
