%%writefile app.py
# --- Voice Biometrics Streamlit App ---
import streamlit as st
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import os, json
from pathlib import Path
from scipy.spatial.distance import cosine
from gtts import gTTS
import tempfile
import plotly.express as px
from sklearn.decomposition import PCA

st.set_page_config(page_title="Voice Biometrics", page_icon="🎤", layout="centered")
st.title("🎤 Voice Biometrics Verification System")

DB_PATH = Path("voice_db.json")
if not DB_PATH.exists():
    with open(DB_PATH, "w") as f:
        json.dump({}, f)

encoder = VoiceEncoder()

# Utility: load & save voice DB
def load_db():
    with open(DB_PATH, "r") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

# Sidebar Mode
mode = st.sidebar.radio("Choose Mode", ["🧠 Register Voice", "🔍 Verify Speaker", "📊 Visualize Embeddings"])

# -----------------------------
# 🧠 Register Voice
# -----------------------------
if mode == "🧠 Register Voice":
    name = st.text_input("Enter your name:")
    uploaded_file = st.file_uploader("Upload your voice sample (.wav)", type=["wav"])
    if st.button("Register Voice"):
        if name and uploaded_file:
            wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(wav_path, "wb") as f:
                f.write(uploaded_file.read())
            wav = preprocess_wav(wav_path)
            embedding = encoder.embed_utterance(wav).tolist()

            db = load_db()
            db[name] = embedding
            save_db(db)

            st.success(f"✅ {name} registered successfully!")
            st.audio(wav_path)
        else:
            st.warning("Please provide a name and upload a .wav file.")

# -----------------------------
# 🔍 Verify Speaker
# -----------------------------
elif mode == "🔍 Verify Speaker":
    uploaded_file = st.file_uploader("Upload voice sample for verification (.wav)", type=["wav"])
    if st.button("Verify Speaker"):
        if uploaded_file:
            wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
            with open(wav_path, "wb") as f:
                f.write(uploaded_file.read())
            wav = preprocess_wav(wav_path)
            embedding = encoder.embed_utterance(wav)

            db = load_db()
            results = []
            for user, ref_emb in db.items():
                sim = 1 - cosine(embedding, ref_emb)
                results.append((user, sim))
            
            results = sorted(results, key=lambda x: x[1], reverse=True)
            st.subheader("Similarity Scores")
            st.dataframe(results, use_container_width=True)

            if results and results[0][1] > 0.80:
                user, score = results[0]
                st.success(f"✅ Verified as {user} (Similarity: {score:.2f})")
                tts = gTTS(f"Welcome back, {user}!")
            else:
                st.error("❌ Speaker not recognized.")
                tts = gTTS("Access denied.")
            
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3")
        else:
            st.warning("Please upload a voice sample.")

# -----------------------------
# 📊 Visualization
# -----------------------------
else:
    db = load_db()
    if len(db) < 2:
        st.warning("Need at least 2 registered voices for visualization.")
    else:
        names = list(db.keys())
        embeddings = np.array(list(db.values()))
        reduced = PCA(n_components=2).fit_transform(embeddings)
        fig = px.scatter(x=reduced[:,0], y=reduced[:,1], text=names, title="Voice Embedding Clusters")
        st.plotly_chart(fig)
