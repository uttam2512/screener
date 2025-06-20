# resume_parser.py

import os
import pdfplumber
import docx
import textract

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_doc(file_path):
    text = textract.process(file_path)
    return text.decode("utf-8")

def extract_resume_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext == ".doc":
        return extract_text_from_doc(file_path)
    else:
        return "Unsupported file format."

# For testing
if __name__ == "__main__":
    path = "../resumes/uttam_resume.pdf"  # change this to test your own file
    print(extract_resume_text(path))
