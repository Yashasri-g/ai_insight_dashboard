import streamlit as st
import numpy as np
import sounddevice as sd
import wavio
import soundfile as sf
import io, os, json
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from gtts import gTTS

# ───────────────────────────────
st.set_page_config(page_title="AI Voice Biometrics", page_icon="🎙️", layout="wide")
st.title("🎙️ AI Voice Biometric Authentication")

# Create folders and load DB
os.makedirs("voices", exist_ok=True)
DB_PATH = "voice_db.json"
if not os.path.exists(DB_PATH):
    with open(DB_PATH, "w") as f:
        json.dump({}, f)
with open(DB_PATH, "r") as f:
    db = json.load(f)

encoder = VoiceEncoder()

# Helper functions
def save_db():
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

def record_audio(duration=5, fs=16000):
    st.info(f"Recording for {duration} seconds... 🎙️")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    wavio.write("voices/temp.wav", audio, fs, sampwidth=2)
    st.success("Recording finished ✅")
    return open("voices/temp.wav", "rb").read()

def process_audio(file_bytes):
    data, samplerate = sf.read(io.BytesIO(file_bytes))
    wav = preprocess_wav(data)
    return encoder.embed_utterance(wav)

# Sidebar mode
mode = st.sidebar.radio("Choose Mode", ["Register Speaker", "Verify Speaker"])
st.sidebar.markdown("ℹ️ This system uses deep speaker embeddings from **Resemblyzer** to identify voices.")

# ───────────────────────────────
if mode == "Register Speaker":
    st.header("🧠 Register New Speaker")
    name = st.text_input("Enter your name:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎙️ Record Voice (5 sec)"):
            audio_bytes = record_audio(duration=5)
            st.session_state["audio"] = audio_bytes
            st.audio(audio_bytes, format="audio/wav")

    with col2:
        uploaded = st.file_uploader("or Upload a WAV file", type=["wav"])
        if uploaded:
            st.session_state["audio"] = uploaded.read()
            st.audio(st.session_state["audio"], format="audio/wav")

    if name and "audio" in st.session_state:
        if st.button("💾 Save Speaker"):
            embedding = process_audio(st.session_state["audio"])
            db[name] = embedding.tolist()
            save_db()
            st.success(f"✅ Speaker '{name}' registered successfully!")
    else:
        st.info("Please record/upload a voice and enter a name before saving.")

# ───────────────────────────────
elif mode == "Verify Speaker":
    st.header("🔍 Verify Speaker")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🎙️ Record Voice (5 sec)"):
            audio_bytes = record_audio(duration=5)
            st.session_state["test_audio"] = audio_bytes
            st.audio(audio_bytes, format="audio/wav")

    with col2:
        uploaded = st.file_uploader("or Upload a WAV file", type=["wav"])
        if uploaded:
            st.session_state["test_audio"] = uploaded.read()
            st.audio(st.session_state["test_audio"], format="audio/wav")

    if "test_audio" in st.session_state and st.button("🔎 Verify Identity"):
        if not db:
            st.warning("⚠️ No registered users found. Please register first.")
        else:
            test_emb = process_audio(st.session_state["test_audio"])
            similarities = {n: 1 - cosine(test_emb, np.array(e)) for n, e in db.items()}
            best_match = max(similarities, key=similarities.get)
            conf = similarities[best_match]

            if conf > 0.75:
                msg = f"✅ Welcome back, {best_match}! (confidence: {conf:.2f})"
                st.success(msg)
            else:
                msg = "❌ Access Denied. Speaker not recognized."
                st.error(msg)

            # Text-to-speech
            tts = gTTS(msg)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            st.sidebar.markdown("### 🧾 Verification Summary")
            st.sidebar.write(f"**Speaker:** {best_match}")
            st.sidebar.write(f"**Confidence:** {conf:.2f}")
