import os
import docx
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer, util
import pandas as pd

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + good enough

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_resume_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        return ""

def get_keywords_from_jd(jd_text):
    doc = nlp(jd_text)
    return list(set([chunk.text.lower().strip() for chunk in doc.noun_chunks]))

def cosine_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0][0])

def match_skills(resume_text, jd_keywords):
    resume_words = set(resume_text.lower().split())
    matched = [skill for skill in jd_keywords if skill in resume_words]
    missing = [skill for skill in jd_keywords if skill not in resume_words]
    return matched, missing

def process_resumes_for_jd(jd_path, resumes_dir, output_dir):
    # Load JD text
    if jd_path.endswith(".docx"):
        jd_text = extract_text_from_docx(jd_path)
    elif jd_path.endswith(".pdf"):
        jd_text = extract_text_from_pdf(jd_path)
    else:
        print(f"Unsupported JD format: {jd_path}")
        return

    jd_keywords = get_keywords_from_jd(jd_text)
    results = []

    for filename in os.listdir(resumes_dir):
        filepath = os.path.join(resumes_dir, filename)
        text = extract_resume_text(filepath)
        if not text:
            continue

        score = cosine_similarity(jd_text, text)
        matched, missing = match_skills(text, jd_keywords)

        results.append({
            "Resume Filename": filename,
            "Score": round(score * 100, 2),
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing),
            "Approval": "✅ Approved" if score > 60 else "❌ Rejected"
        })

    jd_name = os.path.splitext(os.path.basename(jd_path))[0]
    output_csv = os.path.join(output_dir, f"{jd_name}_results.csv")
    df = pd.DataFrame(results)
    df.sort_values(by="Score", ascending=False, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ {jd_name} results saved to {output_csv}")


if __name__ == "__main__":
    resumes_dir = "../resumes"
    jd_dir = "../job_descriptions"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    for jd_file in os.listdir(jd_dir):
        jd_path = os.path.join(jd_dir, jd_file)
        process_resumes_for_jd(jd_path, resumes_dir, output_dir)

    print("\n📊 All JDs processed.")
