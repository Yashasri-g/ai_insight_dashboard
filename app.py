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
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
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
positive = (df["Sentiment"] == "POSITIVE").sum()
negative = (df["Sentiment"] == "NEGATIVE").sum()
positive_pct = round((positive / total) * 100, 1)
negative_pct = round((negative / total) * 100, 1)

# Display metrics in columns
col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", int(total))
col2.metric("Positive Feedback", f"{positive_pct}% 👍", delta=int(positive))
col3.metric("Negative Feedback", f"{negative_pct}% 👎", delta=-int(negative))

# Plot bar chart
fig, ax = plt.subplots(figsize=(4, 3))
df["Sentiment"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_ylabel("Count")
st.pyplot(fig)

# -------------------------------
# SUMMARIZATION
# -------------------------------
st.markdown("## 🧠 Overall Summary")

all_text = " ".join(df["text"].astype(str).tolist())

with st.spinner("Summarizing insights..."):
    summary = summarizer(
        all_text[:2000],
        max_length=120,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

# Two-column layout
c1, c2 = st.columns([1, 2])

with c1:
    st.markdown("### 🪄 Summary")
    st.success(summary)

with c2:
    st.markdown("### 📊 Sentiment Trend")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    df["Sentiment"].value_counts().plot.pie(
        autopct="%1.1f%%",
        colors=["lightgreen", "salmon"],
        ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers")
