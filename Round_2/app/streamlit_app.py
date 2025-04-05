import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Load models
engagement_model = joblib.load("models/engagement_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
tag_model = joblib.load("models/tag_model.pkl")
mlb = joblib.load("models/mlb.pkl")

st.title("🧠 Medium Article Analyzer")

title = st.text_input("Enter the article title")
text = st.text_area("Paste the article content")
reading_time = st.slider("Reading Time (min)", 1, 20, 5)
weekday = datetime.today().weekday()
title_len = len(title)

if st.button("Analyze"):
    input_text = title + " " + text
    X_tfidf = vectorizer.transform([input_text])
    tag_preds = tag_model.predict(X_tfidf)
    tags = mlb.inverse_transform(tag_preds)[0]
    X_features = pd.DataFrame([[reading_time, title_len, len(tags), weekday]],
                              columns=['reading_time', 'title_len', 'num_tags', 'weekday'])
    engagement = engagement_model.predict(X_features)[0]
    st.markdown(f"### 🔖 Predicted Tags: {', '.join(tags)}")
    st.markdown(f"### 📈 Popularity: {'✅ Likely Popular' if engagement == 1 else '❌ Not Likely Popular'}")