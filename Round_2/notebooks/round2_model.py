# -*- coding: utf-8 -*-
"""
AI-Powered Content Analysis - Round 2 Hackathon Solution.ipynb

This notebook contains the code for the second round of the AI-Powered Content Analysis hackathon.
It builds upon the preprocessed data from the first round and implements tag modeling,
engagement prediction, explainability, keyword extraction, and model saving for
frontend integration.
"""

# STEP 1: Import Libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer

import warnings
warnings.filterwarnings("ignore")

# STEP 2: Load Preprocessed Data (from Round 1)
# Assuming the cleaned CSV file is in the '../data/' directory
df = pd.read_csv("../data/cleaned_medium_articles.csv")

# Display the first few rows of the dataframe to verify loading
print("Loaded Data:")
print(df.head())
print("\n")

# STEP 3: Feature Engineering
df['reading_time'] = df['text'].apply(lambda x: len(str(x).split()) // 200)
df['title_len'] = df['title'].apply(lambda x: len(str(x)))
df['num_tags'] = df['tags'].apply(lambda x: len(eval(x)))
df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek

print("Engineered Features:")
print(df[['reading_time', 'title_len', 'num_tags', 'weekday']].head())
print("\n")

# STEP 4: Tag Modeling (Multi-label Classification)
mlb = MultiLabelBinarizer()
y_tags = mlb.fit_transform(df['tags'].apply(eval))

X_text = df['title'] + " " + df['text']
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X_tfidf = tfidf.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_tags, test_size=0.2, random_state=42)

model_tag = MultiOutputClassifier(LogisticRegression(max_iter=1000))
model_tag.fit(X_train, y_train)
y_pred_tags = model_tag.predict(X_test)

tag_f1_score = f1_score(y_test, y_pred_tags, average="micro")
print("Tag Modeling F1 Score:", tag_f1_score)
print("\n")

# STEP 5: Engagement Prediction (Binary Classification)
df['is_popular'] = df['claps'].apply(lambda x: 1 if x > df['claps'].median() else 0)

features = ['reading_time', 'title_len', 'num_tags', 'weekday']
X_engage = df[features]
y_engage = df['is_popular']

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_engage, y_engage, test_size=0.2, random_state=42)

model_engage = XGBClassifier(tree_method='hist')
params = {'n_estimators': [100, 200], 'max_depth': [4, 6]}
grid = GridSearchCV(model_engage, param_grid=params, scoring='f1', cv=3)
grid.fit(X_train_e, y_train_e)

best_model_engage = grid.best_estimator_
y_pred_engage = best_model_engage.predict(X_test_e)
engagement_accuracy = accuracy_score(y_test_e, y_pred_engage)
engagement_f1_score = f1_score(y_test_e, y_pred_engage)
print("Engagement Accuracy:", engagement_accuracy)
print("Engagement F1 Score:", engagement_f1_score)
print("\n")

# STEP 6: Explainability with SHAP
explainer = shap.Explainer(best_model_engage)
shap_values = explainer(X_test_e[:100])
shap.summary_plot(shap_values, pd.DataFrame(X_test_e[:100], columns=features), show=False)
plt.title("SHAP Summary Plot for Engagement Prediction (First 100 Samples)")
plt.show()
print("\n")

# STEP 7: Keyword Extraction (Simple TF-IDF Method)
def extract_keywords(text, n=10):
    vec = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vec.fit_transform([text])
    scores = zip(vec.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0])
    sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
    return [kw for kw, score in sorted_keywords[:n]]

sample_keywords = extract_keywords(df['text'].iloc[0])
print("Sample Keywords for the first article:", sample_keywords)
print("\n")

# STEP 8: Save Models for Frontend Integration
import os
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model_tag, os.path.join(models_dir, "tag_classifier.pkl"))
joblib.dump(best_model_engage, os.path.join(models_dir, "engagement_model.pkl"))
joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
joblib.dump(mlb, os.path.join(models_dir, "tag_mlb.pkl"))

print("Models saved successfully in the '../models/' directory.")
print("\n")

# STEP 9: Streamlit Frontend Code Snippet (Separate Cell for better organization)
print("""
# To run this Streamlit app, save the following code in a file named 'app.py'
# in the same directory as the saved models ('../models/').
# Then, open your terminal, navigate to that directory, and run: streamlit run app.py

import streamlit as st
import joblib

# Load the saved models and vectorizer
try:
    model = joblib.load("../models/engagement_model.pkl")
    tfidf = joblib.load("../models/tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: Make sure the model files are in the '../models/' directory.")
    st.stop()

st.title("AI Content Popularity Prediction")
st.markdown("Paste the title and content of an article to predict its popularity.")

text_input = st.text_area("Article Title and Content")

if st.button("Predict Popularity"):
    if text_input:
        # Transform the input text using the loaded TF-IDF vectorizer
        X_input = tfidf.transform([text_input])
        # Make the prediction using the loaded engagement model
        pred = model.predict(X_input)
        # Display the prediction
        st.subheader("Prediction:")
        if pred[0] == 1:
            st.success("This article is likely to be popular! 🔥")
        else:
            st.info("This article is less likely to be popular. 🧊")
    else:
        st.warning("Please paste the article title and content.")
""")

print("\n")
print("Streamlit frontend code snippet provided. Save it as 'app.py' and run using Streamlit.")
