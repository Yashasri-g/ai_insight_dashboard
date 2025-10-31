import streamlit as st
from audiorecorder import audiorecorder
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from speechbrain.pretrained import EncoderClassifier
from gtts import gTTS
import numpy as np
import torch, json, os

# Load pretrained model
encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

DB_PATH = "voice_db.json"
THRESHOLD = 0.75

# Sidebar
st.sidebar.title("🎧 Voice Biometric AI System")
st.sidebar.markdown("""
- 🧠 Uses SpeechBrain pretrained model  
- 🎤 Record or verify in-browser  
- 🔒 Speaker identification using embeddings
""")

st.title("🔐 Voice Biometrics Demo")
option = st.radio("Choose mode", ["Register Speaker", "Verify Speaker"])

def save_db(data):
    with open(DB_PATH, "w") as f: json.dump(data, f)

def load_db():
    return json.load(open(DB_PATH)) if os.path.exists(DB_PATH) else {}

# Record
audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    audio.export("temp.wav", format="wav")

    emb = encoder.encode_file("temp.wav").detach().cpu().numpy().flatten()

    db = load_db()

    if option == "Register Speaker":
        name = st.text_input("Enter speaker name:")
        if st.button("Register Voice") and name:
            db[name] = emb.tolist()
            save_db(db)
            st.success(f"✅ Speaker '{name}' registered successfully!")
    elif option == "Verify Speaker":
        if st.button("Verify"):
            if not db:
                st.warning("No registered speakers yet.")
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
