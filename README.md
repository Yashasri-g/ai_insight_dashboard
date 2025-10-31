# AI Insight Dashboard

An interactive Streamlit-based dashboard that extracts insights, sentiment analysis, and summaries from text or CSV data , built entirely with free, open-source models.

---

##  Project Overview

AI Insight Dashboard demonstrates how Natural Language Processing (NLP) can help you quickly understand patterns and emotions in text data such as user feedback, reviews, or social media comments. It combines sentiment analysis, summarization, and visual analytics into a single, easy-to-use interface.

Key use cases:
- Analyzing customer feedback and survey responses
- Summarizing long text corpora (reports, comments)
- Quick sentiment trend checks and exploratory analysis

---

## Features

- Upload a CSV (must include a `text` column) or paste raw text
- Automatic sentiment detection (Positive / Negative / Neutral)
- Concise summaries of text using a pre-trained summarization model
- Interactive visualizations (bar charts, metrics)
- Runs locally using open-source Hugging Face models — no API keys or paid services required

---

## Tech Stack

- Frontend/UI: Streamlit
- Models: Hugging Face Transformers
  - Sentiment: `distilbert-base-uncased-finetuned-sst-2-english`
  - Summarization: `facebook/bart-large-cnn`
- Language: Python 3.11
- Visualization: Streamlit charts, Matplotlib
- Hosting: Streamlit Cloud

---

## Repository Structure

ai_insight_dashboard/
- app.py — Main Streamlit application
- requirements.txt — Python dependencies
- runtime.txt — Python version for Streamlit Cloud
- README.md — Project documentation

---

##  Quick Start (Local)

1. Clone the repo
```bash
git clone https://github.com/Yashasri-g/ai_insight_dashboard.git
cd ai_insight_dashboard
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

4. Use the UI
- Upload a CSV (ensure a `text` column) or paste text manually
- View sentiment counts, charts, and generated summaries

---

## 📄 Sample Data

sample_feedback.csv
```
text
The product is amazing and easy to use.
Customer support was slow to respond.
I love the new design and features!
The update caused some bugs in the system.
```

---

## Notes & Limitations

- The dashboard uses pre-trained models and works best with English text.
- For large datasets, processing can be slow — consider batching inputs or preprocessing.
- Summaries and sentiment predictions depend on model limitations and may not be perfect.

---

## Roadmap / Future Enhancements

- Topic clustering and keyword extraction
- PDF and DOCX document analysis
- Multi-language sentiment support
- Voice input for text analysis
- Improved UI for large dataset exploration

---

## Author

- Yashasri Gudhe  
- AI & Data Science Enthusiast — Building open-source AI tools 
---

## Acknowledgements

- Hugging Face Transformers
- Streamlit
- Open-source developer community

---

## License

This project is open source. Add your preferred license (e.g., MIT) in a LICENSE file.
