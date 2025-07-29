{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "ee6f691a-d633-4eac-a2f8-384a36ced2e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-29 07:48:09.773 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-29 07:48:09.775 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-29 07:48:09.776 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import fitz  # PyMuPDF\n",
    "from docx import Document\n",
    "\n",
    "vectorizer = joblib.load(\"resume_vectorizer.pkl\")\n",
    "model = joblib.load(\"resume_classifier.pkl\")\n",
    "\n",
    "st.set_page_config(page_title=\"Resume Classifier\", layout=\"wide\")\n",
    "st.title(\"📄 Resume Classifier App\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "444982ac-cf00-4d0c-ac08-e4a71ea2bb34",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "76fde141-87ab-4b71-aee3-de15dbfdba59",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
