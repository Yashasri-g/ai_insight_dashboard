import os, json, io
import numpy as np
import streamlit as st
from audiorecorder import audiorecorder
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from gtts import gTTS
import soundfile as sf
from pathlib import Path

# ───────────────────────────────
# INITIAL SETUP
st.set_page_config(page_title="Voice Access Portal", page_icon="🎙️", layout="wide")

# ensure ffmpeg
os.system("apt-get update -y && apt-get install -y ffmpeg")

# directories
Path("data/voices").mkdir(parents=True, exist_ok=True)
DB_PATH = Path("data/embeddings.json")

if DB_PATH.exists():
    with open(DB_PATH, "r") as f:
        registered_speakers = json.load(f)
else:
    registered_speakers = {}

encoder = VoiceEncoder()

# ───────────────────────────────
# SIDEBAR
st.sidebar.markdown("""
# 🏢 Voice Access Portal
> Secure, AI-driven authentication by voice
""")
choice = st.sidebar.radio("Navigate", ["🔐 Register Speaker", "🎤 Verify Speaker"])
st.sidebar.info("Ensure clear microphone audio for best results.")

# utility to save embeddings
def save_db():
    with open(DB_PATH, "w") as f:
        json.dump(registered_speakers, f)

# ───────────────────────────────
# 1️⃣ REGISTER SPEAKER
if choice == "🔐 Register Speaker":
    st.title("🔐 Register a New Speaker")
    st.write("Record your voice (5–10 seconds) and enter your name.")

    reg_audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording", key="reg")

    if len(reg_audio) > 0:
        st.audio(reg_audio.export().read(), format="audio/wav")

        name = st.text_input("Enter your name:")
        if st.button("Register Voice"):
            wav_bytes = io.BytesIO()
            reg_audio.export(wav_bytes, format="wav")
            wav_bytes.seek(0)
            wav_data, sr = sf.read(wav_bytes)
            wav = preprocess_wav(wav_data)
            embedding = encoder.embed_utterance(wav)

            registered_speakers[name] = embedding.tolist()
            save_db()

            # save voice file
            with open(f"data/voices/{name}.wav", "wb") as f:
                f.write(wav_bytes.getbuffer())

            st.success(f"✅ Speaker '{name}' registered successfully!")
            st.balloons()

# ───────────────────────────────
# 2️⃣ VERIFY SPEAKER
elif choice == "🎤 Verify Speaker":
    st.title("🎤 Verify Speaker")
    st.write("Record your voice for verification.")

    test_audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording", key="test")

    if len(test_audio) > 0:
        st.audio(test_audio.export().read(), format="audio/wav")

        wav_bytes = io.BytesIO()
        test_audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)
        wav_data, sr = sf.read(wav_bytes)
        wav = preprocess_wav(wav_data)
        test_embedding = encoder.embed_utterance(wav)

        if registered_speakers:
            similarities = {
                name: 1 - cosine(test_embedding, np.array(emb))
                for name, emb in registered_speakers.items()
            }

            identified_name = max(similarities, key=similarities.get)
            confidence = similarities[identified_name]

            st.markdown(f"### 🔎 Identified as: **{identified_name}** (confidence: {confidence:.2f})")

            if confidence > 0.75:
                message = f"✅ Welcome {identified_name}! Access granted."
                color = "green"
            else:
                message = "🚫 Access Denied. Unrecognized voice."
                color = "red"

            st.markdown(f"<h3 style='color:{color}'>{message}</h3>", unsafe_allow_html=True)

            # Voice feedback
            tts = gTTS(message)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            # Sidebar summary
            st.sidebar.markdown("### 🧠 Verification Summary")
            st.sidebar.write(f"**Speaker:** {identified_name}")
            st.sidebar.write(f"**Confidence:** {confidence:.2f}")
        else:
            st.warning("⚠️ No registered speakers found. Please register first.")
