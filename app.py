import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import Counter
import re

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="📊 AI Insight Dashboard", page_icon="🤖", layout="wide")

st.title("📊 AI Insight Dashboard")
st.caption("Analyze text data using open-source AI models — sentiment, summary & insights powered by Hugging Face.")

# -------------------------------
# SIDEBAR INPUT MODE
# -------------------------------
st.sidebar.header("⚙️ Input Options")
input_mode = st.sidebar.radio("Select Input Type:", ["Upload CSV", "Enter Text Manually"])

if input_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file (must include a column named 'text')", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
            st.stop()
    else:
        st.info("👆 Upload a CSV or switch to text input mode to begin analysis")
        st.stop()

else:
    st.sidebar.write("✍️ Enter one or more paragraphs below (each line = one entry):")
    user_text = st.text_area("Enter your text here:", height=200, placeholder="e.g., The new AI assistant is really helpful!")
    if not user_text.strip():
        st.info("Please enter some text to analyze.")
        st.stop()
    df = pd.DataFrame({"text": [t.strip() for t in user_text.split("\n") if t.strip()]})

# -------------------------------
# DISPLAY DATA PREVIEW
# -------------------------------
st.subheader("🧾 Data Preview")
st.dataframe(df.head())

# -------------------------------
# LOAD MODELS
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

# Sentiment counts
sentiment_counts = df["Sentiment"].value_counts()
positive = sentiment_counts.get("POSITIVE", 0)
negative = sentiment_counts.get("NEGATIVE", 0)
total = len(df)

col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", total)
col2.metric("Positive", f"{(positive/total)*100:.1f}%", delta=positive)
col3.metric("Negative", f"{(negative/total)*100:.1f}%", delta=-negative)

# Bar chart
fig, ax = plt.subplots(figsize=(4, 3))
sentiment_counts.plot(kind="bar", color=["green", "red"], ax=ax)
ax.set_title("Sentiment Distribution")
ax.set_ylabel("Count")
st.pyplot(fig)

# -------------------------------
# TEXT SUMMARIZATION
# -------------------------------
st.markdown("## 🧠 Summary of All Text")

all_text = " ".join(df["text"].astype(str).tolist())
with st.spinner("Generating summary..."):
    summary = summarizer(all_text[:2000], max_length=120, min_length=40, do_sample=False)[0]["summary_text"]

st.success(summary)

# -------------------------------
# KEYWORD EXTRACTION (NEW FEATURE)
# -------------------------------
st.markdown("## 🔑 Top Keywords (Simple Frequency-Based)")

def extract_keywords(text, num_keywords=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common = Counter(words).most_common(num_keywords)
    return pd.DataFrame(common, columns=["Keyword", "Frequency"])

keywords_df = extract_keywords(all_text)
st.table(keywords_df)

# -------------------------------
# SENTIMENT PIE CHART
# -------------------------------
st.markdown("## 📊 Sentiment Ratio")
fig2, ax2 = plt.subplots(figsize=(4, 3))
sentiment_counts.plot.pie(autopct="%1.1f%%", colors=["lightgreen", "salmon"], ax=ax2)
ax2.set_ylabel("")
st.pyplot(fig2)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Hugging Face Transformers | Fully Open Source")
