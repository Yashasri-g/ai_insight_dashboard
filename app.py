import os
import io
import numpy as np
import soundfile as sf
import streamlit as st
from audiorecorder import audiorecorder
from pydub import AudioSegment
import imageio_ffmpeg
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine
from gtts import gTTS

# ──────────────────────────────
# SETUP & CONFIG
# ──────────────────────────────
st.set_page_config(page_title="Voice Biometric Assistant", page_icon="🎙️", layout="wide")

# Ensure ffmpeg works for pydub
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

st.title("🎙️ Voice Biometric Assistant")
st.sidebar.header("Navigation")
choice = st.sidebar.radio("Go to:", ["Register Speaker", "Verify Speaker"])

# Initialize AI encoder & speaker database
encoder = VoiceEncoder()
if "speakers" not in st.session_state:
    st.session_state.speakers = {}

# ──────────────────────────────
# 1️⃣ REGISTER SPEAKER
# ──────────────────────────────
if choice == "Register Speaker":
    st.header("🔐 Register a New Speaker")
    st.write("Press **Start Recording**, speak for 5–10 seconds, then **Stop Recording**.")
    
    audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

    if len(audio) > 0:
        st.audio(audio.export().read(), format="audio/wav")
        name = st.text_input("Enter your name:")
        if name and st.button("Register Voice"):
            # Save recording to memory and generate embedding
            wav_data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            embedding = encoder.embed_utterance(preprocess_wav(wav_data))
            st.session_state.speakers[name] = embedding
            st.success(f"✅ Speaker '{name}' registered successfully!")

    if st.session_state.speakers:
        st.subheader("📋 Registered Speakers")
        st.write(list(st.session_state.speakers.keys()))

# ──────────────────────────────
# 2️⃣ VERIFY SPEAKER
# ──────────────────────────────
elif choice == "Verify Speaker":
    st.header("🎤 Verify Speaker Identity")
    st.write("Press **Start Recording**, say something, then **Stop Recording**.")
    
    audio = audiorecorder("🎙️ Start Recording", "⏹️ Stop Recording")

    if len(audio) > 0:
        st.audio(audio.export().read(), format="audio/wav")
        
        if not st.session_state.speakers:
            st.warning("⚠️ No registered speakers. Please register first.")
        else:
            # Create embedding for test audio
            wav_data = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            test_embedding = encoder.embed_utterance(preprocess_wav(wav_data))
            
            # Compare with each known speaker
            similarities = {
                name: 1 - cosine(test_embedding, emb)
                for name, emb in st.session_state.speakers.items()
            }
            identified_name = max(similarities, key=similarities.get)
            confidence = similarities[identified_name]
            
            st.markdown(f"### 🧭 Identified as: **{identified_name}** (confidence: {confidence:.2f})")
            
            # AI-generated response
            if confidence > 0.7:
                response_text = f"Welcome {identified_name}! Access granted ✅"
            else:
                response_text = "Access denied. Unrecognized speaker 🚫"
            
            st.write("💬 Response:", response_text)
            
            # Convert response to voice
            tts = gTTS(response_text)
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            # Sidebar summary
            st.sidebar.markdown("### Detection Summary")
            st.sidebar.write(f"**Speaker:** {identified_name}")
            st.sidebar.write(f"**Confidence:** {confidence:.2f}")
