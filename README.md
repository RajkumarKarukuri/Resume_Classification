# 📄 Resume Classifier App

This Streamlit app classifies resumes into job roles using multiple machine learning models.

## 🔍 Features

- Upload or paste resume text
- Choose from 5 different models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - SVM (Support Vector Machine)
  - XGBoost
- Instantly predict the job role (e.g., SQL Developer, React Developer)

## 🚀 Try the App

**Live App:** [Click here to open](https://your-app-link.streamlit.app)

## 📂 Files in this repo

| File                  | Description                           |
|-----------------------|---------------------------------------|
| `app.py`              | Main Streamlit application            |
| `*.pkl`               | Trained machine learning models       |
| `resume_vectorizer.pkl` | TF-IDF vectorizer used in training |
| `label_encoder.pkl`   | Encoder for job role labels           |
| `requirements.txt`    | All required packages for deployment  |

## 💻 Local Installation

To run this app locally:

```bash
git clone https://github.com/yourusername/resume-classifier-app.git
cd resume-classifier-app
pip install -r requirements.txt
streamlit run app.py
