import re
from io import StringIO
import streamlit as st
import joblib
import fitz
from docx import Document

# Load shared components
vectorizer = joblib.load("resume_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load all models into a dictionary
models = {
    "Logistic Regression": joblib.load("logistic_model.pkl"),
    "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    "Random Forest": joblib.load("random_forest_model.pkl"),
    "SVM": joblib.load("svm_model.pkl"),
    "XGBoost": joblib.load("xgboost_model.pkl")
}

# App layout
st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("🧠 Multi-Model Resume Classifier")

# Model selector
selected_model_name = st.selectbox("Choose a model for prediction:", list(models.keys()))
selected_model = models[selected_model_name]

# Extractor functions
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Sample skill set (extend as needed)
common_skills = {
    "python", "java", "sql", "react", "node", "aws", "azure", "docker", "linux",
    "git", "html", "css", "javascript", "mongodb", "flask", "django", "excel",
    "power bi", "machine learning", "deep learning", "tensorflow", "pandas", "numpy"
}

def extract_email(text):
    matches = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    return matches[0] if matches else "Not found"

def extract_phone(text):
    matches = re.findall(r'(\+91[-\s]?)?[789]\d{9}', text)
    return matches[0] if matches else "Not found"

def extract_skills(text, skills_set):
    text = text.lower()
    found_skills = [skill for skill in skills_set if skill in text]
    return ", ".join(found_skills) if found_skills else "Not found"


# File upload
uploaded_file = st.file_uploader("Upload resume (.pdf or .docx)", type=["pdf", "docx"])
resume_text = ""

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif ext == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    st.success("Resume text extracted!")

# Text input (manual option)
st.markdown("---")
st.write("Or paste resume text manually:")
manual_text = st.text_area("Paste resume content here:", value=resume_text, height=300)

# Predict
if st.button("🔍 Predict Job Role"):
    if not text_input.strip():
        st.warning("Please enter or upload resume text.")
    else:
        vec = vectorizer.transform([text_input])
        prediction = selected_model.predict(vec)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🎯 Predicted Job Role ({selected_model_name}): **{predicted_label}**")

        # 🔍 Extract fields
        email = extract_email(text_input)
        phone = extract_phone(text_input)
        skills = extract_skills(text_input, common_skills)

        # 🧾 Show in sidebar
        st.sidebar.markdown("### 📋 Extracted Details")
        st.sidebar.write(f"📧 **Email:** {email}")
        st.sidebar.write(f"📱 **Phone:** {phone}")
        st.sidebar.write(f"🛠️ **Skills:** {skills}")

        # 📄 Generate downloadable text
        report = f"""
        Resume Classification Report
        -----------------------------
        Predicted Role: {predicted_label}
        Model Used: {selected_model_name}

        📧 Email: {email}
        📱 Phone: {phone}
        🛠️ Skills: {skills}
        """

        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="resume_report.txt",
            mime="text/plain"
        )
