import streamlit as st
import plotly.express as px
import re
from io import BytesIO
import pandas as pd
import docx
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer, util

# Load NLP & Model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit Config
st.set_page_config(page_title="Smart Resume Screener", layout="wide")
st.title("📄 Smart Resume Screener")

st.sidebar.header("Upload Files")

# Upload Multiple JDs and Resumes
jd_files = st.sidebar.file_uploader(
    "📄 Upload Job Descriptions (.pdf or .docx)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)
resumes = st.sidebar.file_uploader(
    "📁 Upload Resumes (multiple allowed)", 
    type=["pdf", "docx"], 
    accept_multiple_files=True
)

# ----------- 🔧 Helper Functions -----------

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_resume_text(file):
    if file.name.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif file.name.endswith('.docx'):
        return extract_text_from_docx(file)
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

def generate_feedback(score, missing_skills):
    if score > 0.6:
        return "Great match! Resume aligns well with the JD."
    elif not missing_skills:
        return "Improve the structure or clarity of resume content."
    else:
        top_missing = ", ".join(missing_skills[:5])
        return f"Add or highlight skills like: {top_missing}"

def extract_contact_info(text):
    email_match = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phone_match = re.findall(r"\+?\d[\d\s().-]{8,}\d", text)
    email = email_match[0] if email_match else "Not Found"
    phone = phone_match[0] if phone_match else "Not Found"
    return email, phone


# ----------- 📤 File Previews -----------

if jd_files:
    st.subheader("✅ Job Descriptions Uploaded:")
    for file in jd_files:
        st.write(f"• {file.name}")

if resumes:
    st.subheader("📄 Resumes Uploaded:")
    for res in resumes:
        st.write(f"• {res.name}")

# ----------- 🧠 Screening Logic -----------

if jd_files and resumes:
    total_resumes = len(resumes)
    total_jds = len(jd_files)

    st.markdown("### 📊 Summary Dashboard")
    st.markdown(f"- **Total JDs uploaded:** {total_jds}")
    st.markdown(f"- **Total Resumes uploaded:** {total_resumes}")

    all_results = []

    for jd_file in jd_files:
        jd_text = extract_resume_text(jd_file)
        jd_keywords = get_keywords_from_jd(jd_text)
        jd_results = []

        approved_count = 0
        total_score = 0

        for res in resumes:
            res_text = extract_resume_text(res)
            email,phone = extract_contact_info(res_text)
            score = cosine_similarity(jd_text, res_text)
            matched, missing = match_skills(res_text, jd_keywords)
            feedback = generate_feedback(score, missing)


            if score > 0.6:
                approved_count += 1
            total_score += score

            jd_results.append({
                "JD File": jd_file.name,
                "Resume File": res.name,
                "Email": email,
                "Phone": phone,
                "Score": round(score * 100, 2),
                "Matched Skills": ", ".join(matched),
                "Missing Skills": ", ".join(missing),
                "Approval": "✅ Approved" if score > 0.6 else "❌ Rejected",
                "Feedback": feedback
            })

        avg_score = round((total_score / total_resumes) * 100, 2)
        df = pd.DataFrame(jd_results).sort_values(by="Score", ascending=False)
        all_results.append((jd_file.name, df))

        st.markdown(f"#### 📝 JD: `{jd_file.name}`")
        st.markdown(f"- ✅ **Approved Resumes:** {approved_count}")
        st.markdown(f"- 📊 **Average Match Score:** {avg_score}%")
        st.dataframe(df, use_container_width=True)
                # -------- 📊 Visual Charts --------
        st.markdown("#### 📈 Visual Analysis")

        # Bar Chart: Resume Score Distribution
        fig_bar = px.bar(
            df,
            x="Resume File",
            y="Score",
            color="Approval",
            title=f"Resume Scores for {jd_file.name}",
            color_discrete_map={"✅ Approved": "green", "❌ Rejected": "red"}
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{jd_file.name}")

        # Pie Chart: Approval vs Rejection
        approval_counts = df["Approval"].value_counts().reset_index()
        approval_counts.columns = ["Status", "Count"]
        fig_pie = px.pie(
            approval_counts,
            names="Status",
            values="Count",
            title=f"Approval Status Breakdown for {jd_file.name}",
            color_discrete_map={"✅ Approved": "green", "❌ Rejected": "red"}
        )
        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{jd_file.name}")



        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"📥 Download Results for {jd_file.name}",
            csv,
            file_name=f"{jd_file.name}_results.csv",
            mime="text/csv"
        )

    # ---------- 📦 Combined CSV + Overall Summary ----------

    combined_df = pd.concat([df.assign(JD_File=jd_name) for jd_name, df in all_results], ignore_index=True)
    total_approved = combined_df["Approval"].value_counts().get("✅ Approved", 0)
    overall_avg = round(combined_df["Score"].mean(), 2)

    st.markdown("### 📦 Combined Results Summary")
    col1, col2 = st.columns(2)
    col1.metric("✅ Total Approved Resumes", total_approved)
    col2.metric("📊 Overall Avg. Score", f"{overall_avg}%")

    csv_all = combined_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download All Results (Combined CSV)", csv_all, "all_results.csv", "text/csv")
