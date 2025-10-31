import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from gtts import gTTS
from audiorecorder import audiorecorder
import numpy as np
import soundfile as sf
import json, io, os

# ─────────────────────────────
# Setup
st.set_page_config(page_title="AI Voice Biometrics", page_icon="🎙️", layout="wide")
st.title("🎙️ AI Voice Biometrics System")
st.caption("Register or verify your voice using deep learning embeddings.")

encoder = VoiceEncoder()
os.makedirs("voices", exist_ok=True)
db_path = "voice_db.json"
if not os.path.exists(db_path):
    with open(db_path, "w") as f:
        json.dump({}, f)

# Sidebar
st.sidebar.header("🔧 Mode Selection")
mode = st.sidebar.radio("Choose Mode", ["Register Speaker", "Verify Speaker"])

with open(db_path, "r") as f:
    db = json.load(f)

# ─────────────────────────────
# Voice Recorder (Reusable)
def record_voice(label):
    st.markdown(f"### {label}")
    audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")
    if len(audio) > 0:
        st.audio(audio.export().read(), format="audio/wav")
        wav_data = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        return wav_data, sr
    return None, None

# ─────────────────────────────
# Registration
if mode == "Register Speaker":
    st.header("🧠 Register New Speaker")
    name = st.text_input("Enter your name:")
    wav_data, sr = record_voice("Record your voice (5–10 sec)...")

    if wav_data is not None and name:
        wav = preprocess_wav(wav_data.astype(np.float32) / np.max(np.abs(wav_data)))
        embedding = encoder.embed_utterance(wav)
        db[name] = embedding.tolist()
        with open(db_path, "w") as f:
            json.dump(db, f)
        st.success(f"✅ Speaker '{name}' registered successfully!")

# ─────────────────────────────
# Verification
elif mode == "Verify Speaker":
    st.header("🔍 Verify Speaker")
    wav_data, sr = record_voice("Record to verify identity...")

    if wav_data is not None:
        if len(db) == 0:
            st.warning("⚠️ No registered voices. Please register first.")
        else:
            wav = preprocess_wav(wav_data.astype(np.float32) / np.max(np.abs(wav_data)))
            test_emb = encoder.embed_utterance(wav)

            similarities = {n: 1 - cosine(test_emb, np.array(e)) for n, e in db.items()}
            best_match = max(similarities, key=similarities.get)
            conf = similarities[best_match]

            st.markdown(f"### 🧭 Match: **{best_match}**  |  Confidence: `{conf:.2f}`")

            if conf > 0.8:
                text = f"Welcome back {best_match}! Access granted."
                st.success(text)
            else:
                text = "Access denied. Speaker not recognized."
                st.error(text)

            # Voice feedback
            tts = gTTS(text)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            # Sidebar summary
            st.sidebar.markdown("### 📊 Detection Summary")
            st.sidebar.write(f"**Speaker:** {best_match}")
            st.sidebar.write(f"**Confidence:** {conf:.2f}")
