import streamlit as st
import numpy as np
import json
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
import tempfile
import os

# Initialize encoder and database
encoder = VoiceEncoder()
DB_PATH = Path("voice_db.json")

if not DB_PATH.exists():
    with open(DB_PATH, "w") as f:
        json.dump({}, f)

# App title
st.title("🎤 Voice Biometrics Demo")
st.caption("Register your voice and verify identity using speaker embeddings")

# Tabs for Register and Verify
tab1, tab2 = st.tabs(["🧠 Register Voice", "🔍 Verify Speaker"])

# ----------------------- Registration Tab -----------------------
with tab1:
    st.header("🧩 Voice Enrollment")
    name = st.text_input("Enter speaker name:")
    uploaded_file = st.file_uploader("Upload a short WAV file (3–10 sec)", type=["wav"], key="register")

    if uploaded_file and name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        wav = preprocess_wav(Path(tmp_path))
        embedding = encoder.embed_utterance(wav)

        # Save embedding
        with open(DB_PATH, "r+") as f:
            db = json.load(f)
            db[name] = embedding.tolist()
            f.seek(0)
            json.dump(db, f, indent=2)

        st.success(f"✅ Speaker '{name}' registered successfully!")
        st.audio(uploaded_file, format="audio/wav")

# ----------------------- Verification Tab -----------------------
with tab2:
    st.header("🎙️ Speaker Verification")
    test_file = st.file_uploader("Upload a voice sample for verification", type=["wav"], key="verify")

    if test_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(test_file.read())
            tmp_path = tmp.name

        wav_verify = preprocess_wav(Path(tmp_path))
        embedding_verify = encoder.embed_utterance(wav_verify)

        with open(DB_PATH, "r") as f:
            db = json.load(f)

        if not db:
            st.warning("No registered voices yet. Please register first.")
        else:
            scores = {}
            for user, emb_list in db.items():
                emb = np.array(emb_list)
                similarity = 1 - cosine(embedding_verify, emb)
                scores[user] = round(similarity, 3)

            st.subheader("🔎 Similarity Scores")
            st.dataframe(
                {"Speaker": list(scores.keys()), "Similarity": list(scores.values())},
                use_container_width=True,
            )

            best_match = max(scores, key=scores.get)
            if scores[best_match] > 0.80:
                st.success(f"✅ Verified as {best_match} ({scores[best_match]})")
            else:
                st.error("❌ Speaker not recognized.")

            st.audio(test_file, format="audio/wav")
