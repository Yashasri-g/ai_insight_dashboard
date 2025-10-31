import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="📊 AI Insight Dashboard", page_icon="🤖", layout="wide")

st.title("📊 AI Insight Dashboard")
st.caption("Analyze text data using open-source AI models for sentiment, summaries & insights")

# -------------------------------
# FILE OR TEXT INPUT
# -------------------------------
st.sidebar.header("📂 Data Input")
input_mode = st.sidebar.radio("Select Input Type", ["📄 Upload CSV", "📝 Enter Text Manually"])

if input_mode == "📄 Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must have a 'text' column)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a column named 'text'")
            st.stop()
    else:
        st.info("👆 Upload a CSV to begin analysis")
        st.stop()

else:
    user_text = st.text_area("🧠 Enter your text or comments (each line = one entry):", height=200)
    if not user_text.strip():
        st.info("✍️ Enter some text to analyze")
        st.stop()
    df = pd.DataFrame({"text": user_text.split("\n")})

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
# SENTIMENT ANALYSIS
# -------------------------------
st.markdown("## 💬 Sentiment Insights")

with st.spinner("Analyzing sentiments..."):
    df["Sentiment"] = df["text"].apply(lambda x: sentiment_analyzer(x[:512])[0]["label"])

# Sentiment metrics
total = len(df)
positive = int((df["Sentiment"] == "POSITIVE").sum())
negative = int((df["Sentiment"] == "NEGATIVE").sum())
positive_pct = round((positive / total) * 100, 1)
negative_pct = round((negative / total) * 100, 1)

col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", str(total))
col2.metric("Positive", f"{positive_pct}%", delta=str(positive))
col3.metric("Negative", f"{negative_pct}%", delta=str(-negative))

# Sentiment Distribution
fig, ax = plt.subplots(figsize=(4, 3))
df["Sentiment"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_ylabel("Count")
st.pyplot(fig)

# -------------------------------
# SUMMARIZATION
# -------------------------------
st.markdown("## 🧠 Overall Summary")

combined_text = " ".join(df["text"].astype(str).tolist())
with st.spinner("Summarizing insights..."):
    summary = summarizer(combined_text[:2000], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]

colA, colB = st.columns([1, 2])
with colA:
    st.markdown("### 🪄 AI Summary")
    st.success(summary)
with colB:
    st.markdown("### 📊 Sentiment Share")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    df["Sentiment"].value_counts().plot.pie(autopct="%1.1f%%", colors=["lightgreen", "salmon"], ax=ax2)
    ax2.set_ylabel("")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers")
