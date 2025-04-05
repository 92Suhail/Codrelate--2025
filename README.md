# 🧠 AI-Powered Content Analysis & Recommendation System

This repository presents an end-to-end solution developed for a hackathon challenge (Round 2). The project combines Machine Learning (ML), Natural Language Processing (NLP), and model explainability (XAI) to analyze Medium articles and provide intelligent content tagging, popularity prediction, and keyword extraction through an interactive frontend.

---

## 📌 Project Highlights

- **🎯 Tag Prediction**: Automatically predicts relevant tags for Medium articles.
- **📈 Engagement Forecasting**: Classifies articles as "popular" or "not popular".
- **🗝️ Keyword Extraction**: Extracts informative keywords from articles using TF-IDF.
- **🔍 Explainability**: SHAP analysis to explain model predictions.
- **💻 Frontend**: Interactive Streamlit app for real-time analysis.

---


---

## 🔧 Technologies Used

- Python (3.8+)
- Pandas, NumPy, scikit-learn
- XGBoost, SHAP, TF-IDF
- Streamlit (for frontend)
- Joblib (for model persistence)

---

🔍 Problem Definition
The core objectives of the system are:

Tag Modeling (Multi-label classification): Predicts multiple relevant tags per article using Logistic Regression and TF-IDF.

Engagement Prediction (Binary classification): Predicts whether an article will be popular using XGBoost.

Keyword Extraction (Unsupervised NLP): Identifies key terms summarizing the article using TF-IDF.

Frontend Deployment: Allows users to input article content and get instant analysis.

⚙️ Feature Engineering
reading_time: Words / 200 WPM

title_len: Length of the article title

num_tags: Count of assigned tags

weekday: Extracted from publication timestamp

📊 Model Performance
Task	Model	Metric	Score
Tag Prediction	Logistic Regression	F1-Score	~0.69
Engagement Prediction	XGBoost	Accuracy	~0.76
F1-Score	~0.78
🧠 Explainability
SHAP was used to interpret the XGBoost engagement prediction model.

Key Features:

Higher reading_time and more num_tags positively impact popularity.

Extremely short titles often result in lower engagement.

📷 Frontend Overview
The Streamlit app enables users to:

Input article title and text

View predicted popularity

See extracted keywords

More features like real-time tag prediction and SHAP visualizations can be added.

🧪 Future Enhancements
Integrate Transformer models (e.g., BERT) for semantic tag prediction.

Add author-level metrics for influence analysis.

Implement collaborative filtering for personalized recommendations.

Expand frontend with tag suggestions, SHAP dashboards, and article similarity.

📁 Deliverables
round2_model.ipynb: Full modeling pipeline

streamlit_app.py: Functional frontend

models/: Serialized trained models

report.pdf: Structured project report

README.md: Project overview & documentation

🏆 Hackathon Alignment
✅ Problem Understanding
✅ Feature Engineering
✅ Accuracy & Optimization
✅ Frontend Deployment
✅ Explainability with SHAP
✅ Professional Report & Code

🙋 Contact
For queries or collaboration: suhailraza0555@gmail.com/ sudipkotal2003@gmail.com

