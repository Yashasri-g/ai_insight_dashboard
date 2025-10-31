import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="📊 AI Insight Dashboard", page_icon="🤖", layout="wide")

st.title("📊 AI Insight Dashboard")
st.caption("Analyze text data using open-source AI models for sentiment & summaries")

# -------------------------------
# FILE UPLOAD SECTION
# -------------------------------
st.sidebar.header("📂 Upload Your Text Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file (with a column named 'text')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
        st.stop()
else:
    st.info("👆 Upload a CSV to begin analysis")
    st.stop()

# Display data preview
st.subheader("🧾 Data Preview")
st.dataframe(df.head())

st.success("✅ File loaded successfully! Next, we’ll add AI insights...")
