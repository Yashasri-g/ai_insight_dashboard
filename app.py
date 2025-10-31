import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

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

st.subheader("🧾 Data Preview")
st.dataframe(df.head())

# -------------------------------
# LOAD MODELS (cached)
# -------------------------------
@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return sentiment_analyzer, summarizer

sentiment_analyzer, summarizer = load_models()

# -------------------------------
# RUN SENTIMENT ANALYSIS
# -------------------------------
st.subheader("💬 Sentiment Analysis")
with st.spinner("Analyzing sentiments..."):
    df["Sentiment"] = df["text"].apply(lambda x: sentiment_analyzer(x[:512])[0]["label"])

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

# Plot chart
fig, ax = plt.subplots()
sentiment_counts.plot(kind="bar", color=["green", "red"], ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_ylabel("Count")
st.pyplot(fig)

# -------------------------------
# GENERATE SUMMARY
# -------------------------------
st.subheader("🧠 AI Summary of All Feedback")
all_text = " ".join(df["text"].astype(str).tolist())

with st.spinner("Summarizing insights..."):
    summary = summarizer(all_text[:2000], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]

st.success(summary)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers")
