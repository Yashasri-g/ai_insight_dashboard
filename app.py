import streamlit as st
from streamlit_audio_recorder import st_audio_recorder
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import numpy as np
import json, os, io

# ------------------- SETUP -------------------
st.set_page_config(page_title="AI Voice Biometrics", page_icon="🎙️", layout="wide")
st.title("🎧 AI Voice Biometrics System")

# Sidebar
st.sidebar.header("Voice AI System")
st.sidebar.info("🔐 Register or Verify your identity by voice.")

DB_PATH = "voice_db.json"
THRESHOLD = 0.75

# Load AI Model
@st.cache_resource
def load_encoder():
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

encoder = load_encoder()

# Database helpers
def load_db():
    return json.load(open(DB_PATH)) if os.path.exists(DB_PATH) else {}

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

# ------------------- RECORD SECTION -------------------
mode = st.radio("Choose Mode", ["Register Speaker", "Verify Speaker"])
st.write("🎙️ Press below to record and stop:")

audio_bytes = st_audio_recorder(key="recorder")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    # Get embedding
    emb = encoder.encode_file("temp.wav").detach().cpu().numpy().flatten()

    db = load_db()

    if mode == "Register Speaker":
        name = st.text_input("Enter your name:")
        if st.button("Register") and name:
            db[name] = emb.tolist()
            save_db(db)
            st.success(f"✅ Speaker '{name}' registered successfully!")

    elif mode == "Verify Speaker":
        if st.button("Verify"):
            if not db:
                st.warning("⚠️ No registered speakers. Please register first.")
            else:
                sims = {n: cosine_similarity([emb], [np.array(v)])[0][0] for n, v in db.items()}
                best_name = max(sims, key=sims.get)
                conf = sims[best_name]
                if conf > THRESHOLD:
                    msg = f"Welcome back, {best_name}!"
                    st.success(f"🎯 {msg} (confidence={conf:.2f})")
                else:
                    msg = "Access denied. Speaker not recognized."
                    st.error(msg)

                tts = gTTS(msg)
                tts.save("response.mp3")
                st.audio("response.mp3", format="audio/mp3")
