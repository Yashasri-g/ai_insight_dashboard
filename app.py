import os
# Ensure ffmpeg is available for pydub / audio decoding
os.system("apt-get update -y && apt-get install -y ffmpeg")

import streamlit as st
from audiorecorder import audiorecorder
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import numpy as np
import io
import soundfile as sf
from gtts import gTTS

# ───────────────────────────────────────────────
# STREAMLIT SETUP
st.set_page_config(page_title="Voice Biometric Assistant", page_icon="🎙️", layout="wide")

st.sidebar.title("🎧 Voice Biometric Assistant")
st.sidebar.write("""
### About this App
- 🎙️ Record your voice in real time  
- 🧠 Identify who you are  
- 💬 Get a text + audio response
""")

encoder = VoiceEncoder()
registered_speakers = {}

# ───────────────────────────────────────────────
# SECTION: Speaker Registration
st.header("🔐 Register Speaker")
st.write("Press **Start Recording**, say a few words (5–10 seconds), then **Stop Recording** and enter your name to register.")

reg_audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording", key="reg")

if len(reg_audio) > 0:
    st.audio(reg_audio.export().read(), format="audio/wav")

    name = st.text_input("Enter your name:", key="name")
    if st.button("Register Voice"):
        # Convert pydub AudioSegment to NumPy wav
        wav_bytes = io.BytesIO()
        reg_audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        wav_data, sr = sf.read(wav_bytes)
        wav = preprocess_wav(wav_data)
        embedding = encoder.embed_utterance(wav)

        registered_speakers[name] = embedding
        st.success(f"✅ Speaker '{name}' registered successfully!")

# ───────────────────────────────────────────────
# SECTION: Speaker Verification
st.header("🎤 Identify Speaker")
st.write("Now record your voice again — the system will try to recognize you.")

test_audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording", key="test")

if len(test_audio) > 0:
    st.audio(test_audio.export().read(), format="audio/wav")

    # Convert pydub AudioSegment to NumPy wav
    wav_bytes = io.BytesIO()
    test_audio.export(wav_bytes, format="wav")
    wav_bytes.seek(0)
    wav_data, sr = sf.read(wav_bytes)
    wav = preprocess_wav(wav_data)
    test_embedding = encoder.embed_utterance(wav)

    if registered_speakers:
        similarities = {
            name: 1 - cosine(test_embedding, emb)
            for name, emb in registered_speakers.items()
        }
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
