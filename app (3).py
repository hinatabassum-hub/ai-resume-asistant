import io
import json
import os
import re
import zipfile
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET

import streamlit as st
import requests
from pypdf import PdfReader

MODEL_NAME = "gemini-2.5-flash"
MAX_FILE_MB = 10
MAX_RESUME_CHARS = 45000


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def extract_docx_text(file_bytes: bytes) -> str:
    """Extract DOCX text using only Python's standard library."""
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        text = "".join(
            node.text or "" for node in paragraph.findall(".//w:t", ns)
        ).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs).strip()


def extract_txt_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_resume_text(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return extract_pdf_text(data)
    if ext == "docx":
        return extract_docx_text(data)
    if ext == "txt":
        return extract_txt_text(data)
    raise ValueError("Supported formats are PDF, DOCX, and TXT.")


def get_api_key() -> Optional[str]:
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key else None


def local_checks(resume: str, job_description: str) -> Dict[str, Any]:
    lower = resume.lower()
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]*\b", resume)
    section_patterns = {
        "Contact Information": r"(email|e-mail|phone|mobile|linkedin|github)",
        "Summary / Profile": r"(professional summary|summary|profile|objective)",
        "Experience": r"(work experience|professional experience|employment|experience)",
        "Education": r"(education|academic)",
        "Skills": r"(skills|technical skills|core competencies|technologies)",
        "Projects": r"(projects|personal projects|academic projects)",
    }
    found = [name for name, pattern in section_patterns.items() if re.search(pattern, lower)]
    bullets = sum(1 for line in resume.splitlines() if re.match(r"^\s*[-•*▪◦]\s+", line))
    verbs = {
        "achieved", "analyzed", "automated", "built", "created", "designed",
        "developed", "delivered", "implemented", "improved", "increased",
        "launched", "led", "managed", "optimized", "reduced", "resolved",
        "streamlined", "tested", "trained", "coordinated",
    }
    action_count = sum(1 for w in re.findall(r"\b[a-zA-Z]+\b", lower) if w in verbs)
    metric_count = len(re.findall(
        r"(\b\d+(?:\.\d+)?\s*%|\$\s?\d+(?:[,.]\d+)*|\b\d+(?:\.\d+)?\s*(?:k|m|million|billion|years?|months?))",
        lower, flags=re.I
    ))
    matched = []
    total = 0
    if job_description.strip():
        jd_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#./-]{2,}\b", job_description.lower())
        stop = {
            "the", "and", "for", "with", "from", "that", "this", "are", "you",
            "your", "our", "will", "have", "has", "had", "was", "were", "job",
            "role", "work", "years", "year", "into", "about", "they", "their",
            "who", "but", "not", "can", "all", "using", "use", "required",
            "requirements", "candidate", "looking", "team", "including", "ability",
        }
        unique = []
        seen = set()
        for word in jd_words:
            if word not in stop and word not in seen:
                seen.add(word)
                unique.append(word)
        total = len(unique)
        matched = [word for word in unique if word in lower]
    return {
        "word_count": len(words),
        "found_sections": found,
        "missing_sections": [x for x in section_patterns if x not in found],
        "bullet_count": bullets,
        "action_verb_count": action_count,
        "metric_count": metric_count,
        "keyword_match_count": len(matched),
        "keyword_total": total,
        "matched_keywords": matched[:40],
    }


def analyze_resume(resume: str, job_description: str, checks: Dict[str, Any]) -> Dict[str, Any]:
    """Call Gemini directly over HTTPS, avoiding the google-genai SDK."""
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it in Streamlit Cloud: "
            "Manage app -> Settings -> Secrets."
        )

    schema = {
        "type": "OBJECT",
        "properties": {
            "ats_score": {"type": "INTEGER", "minimum": 0, "maximum": 100},
            "score_breakdown": {
                "type": "OBJECT",
                "properties": {
                    "format_structure": {"type": "INTEGER", "minimum": 0, "maximum": 100},
                    "keyword_alignment": {"type": "INTEGER", "minimum": 0, "maximum": 100},
                    "experience_impact": {"type": "INTEGER", "minimum": 0, "maximum": 100},
                    "skills_completeness": {"type": "INTEGER", "minimum": 0, "maximum": 100},
                },
                "required": ["format_structure", "keyword_alignment", "experience_impact", "skills_completeness"],
            },
            "summary": {"type": "STRING"},
            "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
            "improvements": {"type": "ARRAY", "items": {"type": "STRING"}},
            "missing_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
            "rewrite_examples": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["ats_score", "score_breakdown", "summary", "strengths", "improvements", "missing_keywords", "rewrite_examples"],
    }

    prompt = f"""
You are an expert ATS resume reviewer and career assistant.

Analyze the resume and return an estimated ATS-readiness score from 0 to 100.
This is not a score from a specific employer ATS and is not a hiring prediction.

Score these dimensions: format/structure, keyword alignment, experience/impact,
and skills/completeness. Never invent experience, education, skills, employers,
dates, achievements, certifications, or numbers. Suggest relevant keywords from
the job description. If no job description is supplied, say keyword alignment is limited.
Give specific improvements and rewrite examples based only on information present.

Deterministic checks:
{json.dumps(checks, indent=2)}

Job description:
{job_description.strip() if job_description.strip() else "(Not provided)"}

Resume:
--- BEGIN RESUME ---
{resume[:MAX_RESUME_CHARS]}
--- END RESUME ---

Return only valid JSON matching the supplied schema.
"""

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
    )

    try:
        response = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=90,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not connect to Gemini API: {exc}") from exc

    if response.status_code != 200:
        try:
            body = response.json()
            message = body.get("error", {}).get("message", str(body))
        except Exception:
            message = response.text[:1000]
        raise RuntimeError(f"Gemini API error ({response.status_code}): {message}")

    try:
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini returned an unexpected or invalid JSON response.") from exc

    result["ats_score"] = max(0, min(100, int(result.get("ats_score", 0))))
    return result


st.set_page_config(page_title="AI Resume Assistant", page_icon="📄", layout="wide")
st.title("📄 AI Resume Assistant")
st.write("Upload a resume, optionally paste a job description, and get an estimated ATS score and practical improvements.")

with st.sidebar:
    st.header("How it works")
    st.markdown("1. Upload resume\n2. Paste job description\n3. Click Analyze Resume\n4. Review ATS score and recommendations")
    st.caption(f"AI model: {MODEL_NAME}")
    st.caption("Gemini is called through HTTPS; no Google GenAI SDK is required.")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Job Description (recommended)", height=220, placeholder="Paste the job description here for better keyword matching.")

if uploaded_file:
    if uploaded_file.size / (1024 * 1024) > MAX_FILE_MB:
        st.error(f"File is too large. Maximum size is {MAX_FILE_MB} MB.")
        st.stop()
    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting resume text..."):
                resume_text = extract_resume_text(uploaded_file)
            if len(resume_text.strip()) < 100:
                st.error("Not enough text could be extracted. Use a text-based PDF or DOCX.")
                st.stop()
            resume_text = resume_text[:MAX_RESUME_CHARS]
            checks = local_checks(resume_text, job_description)
            with st.spinner("Gemini is reviewing your resume..."):
                analysis = analyze_resume(resume_text, job_description, checks)
            st.session_state["analysis"] = analysis
            st.session_state["checks"] = checks
        except Exception as exc:
            st.error("The resume could not be analyzed.")
            st.code(str(exc))

if "analysis" in st.session_state:
    analysis = st.session_state["analysis"]
    checks = st.session_state["checks"]
    st.divider()
    score = analysis["ats_score"]
    status = "Excellent ATS readiness" if score >= 85 else "Good ATS readiness" if score >= 70 else "Needs improvement" if score >= 50 else "Major improvements recommended"
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Estimated ATS Score", f"{score}/100")
        st.progress(score / 100)
        st.caption(status)
    with c2:
        st.subheader("Resume checks")
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Words", checks["word_count"])
        x2.metric("Sections", len(checks["found_sections"]))
        x3.metric("Bullets", checks["bullet_count"])
        x4.metric("Metrics", checks["metric_count"])

    st.subheader("📊 Score Breakdown")
    b = analysis["score_breakdown"]
    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Format & Structure", f"{b['format_structure']}/100")
    x2.metric("Keyword Alignment", f"{b['keyword_alignment']}/100")
    x3.metric("Experience & Impact", f"{b['experience_impact']}/100")
    x4.metric("Skills & Completeness", f"{b['skills_completeness']}/100")

    st.subheader("🧠 Overall Assessment")
    st.write(analysis["summary"])
    left, right = st.columns(2)
    with left:
        st.subheader("✅ Strengths")
        for item in analysis["strengths"]:
            st.markdown(f"- {item}")
    with right:
        st.subheader("🚀 Improvements")
        for item in analysis["improvements"]:
            st.markdown(f"- {item}")

    st.subheader("🔑 Missing / Recommended Keywords")
    if analysis["missing_keywords"]:
        for keyword in analysis["missing_keywords"]:
            st.markdown(f"- `{keyword}`")
    else:
        st.success("No major missing keywords were identified.")

    st.subheader("✍️ Rewrite Examples")
    for example in analysis["rewrite_examples"]:
        st.markdown(f"- {example}")

    if checks["missing_sections"]:
        st.subheader("📌 Sections That May Be Missing")
        for section in checks["missing_sections"]:
            st.markdown(f"- {section}")

    st.info("ATS scores vary by employer, ATS vendor, job description, and resume format. Treat this score as an optimization guide, not a hiring prediction.")
