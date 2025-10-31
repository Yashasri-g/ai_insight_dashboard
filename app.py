import streamlit as st
from streamlit_audiorecorder import audiorecorder
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from gtts import gTTS
import numpy as np
import soundfile as sf
import io, json, os

# ─────────────────────────────
# Page setup
st.set_page_config(page_title="AI Voice Biometrics", page_icon="🎙️", layout="wide")
st.title("🎙️ AI Voice Biometrics System")
st.caption("Register and verify users by voice using deep learning embeddings.")

encoder = VoiceEncoder()
os.makedirs("voices", exist_ok=True)
DB_PATH = "voice_db.json"
if not os.path.exists(DB_PATH):
    with open(DB_PATH, "w") as f:
        json.dump({}, f)

with open(DB_PATH, "r") as f:
    db = json.load(f)

st.sidebar.header("🔧 Mode Selection")
mode = st.sidebar.radio("Choose Mode", ["Register Speaker", "Verify Speaker"])

# ─────────────────────────────
# Helper
def save_db():
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

def process_audio(audio_bytes):
    data, samplerate = sf.read(io.BytesIO(audio_bytes))
    wav = preprocess_wav(data)
    return encoder.embed_utterance(wav)

# ─────────────────────────────
# Record voice (real-time)
def record_audio_ui(label):
    st.markdown(f"### {label}")
    audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")
    if len(audio) > 0:
        st.audio(audio.tobytes(), format="audio/wav")
        return audio.tobytes()
    return None

# ─────────────────────────────
# Registration
if mode == "Register Speaker":
    st.header("🧠 Register New Speaker")
    name = st.text_input("Enter your name:")
    audio_bytes = record_audio_ui("Record your voice (5–10 sec)...")

    if audio_bytes and name:
        embedding = process_audio(audio_bytes)
        db[name] = embedding.tolist()
        save_db()
        st.success(f"✅ Speaker '{name}' registered successfully!")

# ─────────────────────────────
# Verification
elif mode == "Verify Speaker":
    st.header("🔍 Verify Speaker")
    audio_bytes = record_audio_ui("Record to verify identity...")

    if audio_bytes:
        if len(db) == 0:
            st.warning("⚠️ No registered voices. Please register first.")
        else:
            test_emb = process_audio(audio_bytes)
            similarities = {n: 1 - cosine(test_emb, np.array(e)) for n, e in db.items()}
            best_match = max(similarities, key=similarities.get)
            conf = similarities[best_match]

            st.markdown(f"### 🧭 Match: **{best_match}** | Confidence: `{conf:.2f}`")

            if conf > 0.8:
                text = f"Welcome back {best_match}! Access granted."
                st.success(text)
            else:
                text = "Access denied. Speaker not recognized."
                st.error(text)

            # 🎧 Voice feedback
            tts = gTTS(text)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            # Sidebar summary
            st.sidebar.markdown("### 📊 Detection Summary")
            st.sidebar.write(f"**Speaker:** {best_match}")
            st.sidebar.write(f"**Confidence:** {conf:.2f}")
