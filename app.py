import streamlit as st
import joblib
import fitz  # PyMuPDF
from docx import Document

# Load vectorizer and model
vectorizer = joblib.load("resume_vectorizer.pkl")
model = joblib.load("resume_classifier.pkl")

st.set_page_config(page_title="Resume Classifier", layout="wide")
st.title("📄 Resume Classifier App")

# Functions to extract text
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# File Upload
uploaded_file = st.file_uploader("📤 Upload Resume (.pdf or .docx)", type=["pdf", "docx"])
resume_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_type == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("❌ Unsupported file type.")

    st.success("✅ Resume text extracted!")

# Optional: manual text input
st.markdown("---")
st.write("Or paste your resume content here 👇")
text_input = st.text_area("✍️ Paste Resume Text Here", value=resume_text, height=300)

# Prediction
if st.button("🔍 Predict Job Role"):
    if not text_input.strip():
        st.warning("Please enter or upload resume text.")
    else:
        vec = vectorizer.transform([text_input])
        prediction = model.predict(vec)[0]
        st.success(f"🧠 Predicted Job Role: **{prediction}**")
