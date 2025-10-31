import streamlit as st
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from gtts import gTTS
import numpy as np
import soundfile as sf
import io, json, os, base64

# ─────────────────────────────────────
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

# ─────────────────────────────────────
# Helper functions
def save_db():
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

def process_audio(file_bytes):
    data, samplerate = sf.read(io.BytesIO(file_bytes))
    wav = preprocess_wav(data)
    return encoder.embed_utterance(wav)

def record_audio_ui(label):
    st.markdown(f"### {label}")
    record_script = """
        <script>
        const startBtn = document.createElement("button");
        startBtn.textContent = "🎙️ Start Recording";
        startBtn.style = "margin: 8px; padding: 8px; font-size: 16px;";
        const stopBtn = document.createElement("button");
        stopBtn.textContent = "⏹️ Stop Recording";
        stopBtn.style = "margin: 8px; padding: 8px; font-size: 16px;";
        const container = document.currentScript.parentElement;
        container.appendChild(startBtn);
        container.appendChild(stopBtn);
        let recorder, audioChunks;
        startBtn.onclick = async () => {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            recorder = new MediaRecorder(stream);
            recorder.ondataavailable = e => audioChunks.push(e.data);
            recorder.start();
        };
        stopBtn.onclick = async () => {
            recorder.stop();
            recorder.onstop = async () => {
                const blob = new Blob(audioChunks, { type: "audio/wav" });
                const arrayBuffer = await blob.arrayBuffer();
                const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                const pyCode = "st.session_state['audio_base64'] = '" + base64 + "'";
                fetch("/_stcore/execute", {
                    method: "POST",
                    body: JSON.stringify({ code: pyCode }),
                    headers: {"Content-Type": "application/json"},
                });
            };
        };
        </script>
    """
    st.components.v1.html(record_script, height=100)
    if "audio_base64" in st.session_state:
        audio_bytes = base64.b64decode(st.session_state.pop("audio_base64"))
        st.audio(audio_bytes, format="audio/wav")
        return audio_bytes
    return None

# ─────────────────────────────────────
# Registration
if mode == "Register Speaker":
    st.header("🧠 Register New Speaker")
    name = st.text_input("Enter your name:")
    st.markdown("#### Record or upload your voice (5–10 seconds)")
    audio_bytes = record_audio_ui("🎧 Real-time recorder")
    uploaded = st.file_uploader("or upload a WAV file", type=["wav"])
    if uploaded:
        audio_bytes = uploaded.read()

    if audio_bytes and name:
        embedding = process_audio(audio_bytes)
        db[name] = embedding.tolist()
        save_db()
        st.success(f"✅ Speaker '{name}' registered successfully!")

# ─────────────────────────────────────
# Verification
elif mode == "Verify Speaker":
    st.header("🔍 Verify Speaker")
    st.markdown("#### Record or upload voice for verification")
    audio_bytes = record_audio_ui("🎧 Real-time recorder")
    uploaded = st.file_uploader("or upload a WAV file", type=["wav"])
    if uploaded:
        audio_bytes = uploaded.read()

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

            tts = gTTS(text)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            st.sidebar.markdown("### 📊 Detection Summary")
            st.sidebar.write(f"**Speaker:** {best_match}")
            st.sidebar.write(f"**Confidence:** {conf:.2f}")
