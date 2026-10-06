"""
AI Resume & Job Description Analyzer - Version 2 (Streamlit App)
================================================================
A beginner-friendly web application combining deterministic keyword matching (V1)
with AI-powered contextual resume analysis using Google's Gemini API (V2).
"""

import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from main import (
    PREDEFINED_SKILLS,
    extract_text_from_pdf,
    clean_text,
    extract_skills,
    calculate_match,
    analyze_keywords,
    generate_suggestions,
    analyze_resume_and_job,
    SAMPLE_JOB_DESCRIPTION
)

# ------------------------------------------------------------------------------
# Configuration Constants
# ------------------------------------------------------------------------------
# Store the currently supported Gemini Flash model in ONE configuration constant
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"


# ------------------------------------------------------------------------------
# Gemini API Integration Helper (V2)
# ------------------------------------------------------------------------------
def get_gemini_api_key() -> str:
    """
    Safely retrieves the Gemini API key from the local .env file or environment.
    Never exposes, prints, or displays the key.
    """
    # Look for .env in the project directory first, then standard search
    project_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(project_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(find_dotenv(), override=True)

    key = os.environ.get("GEMINI_API_KEY", "")
    return key.strip() if key else ""


def generate_gemini_analysis(
    resume_text: str,
    jd_text: str,
    match_score: float,
    matching_skills: list[str],
    missing_skills: list[str],
    missing_keywords: list[tuple[str, int]],
    model_name: str = GEMINI_MODEL_NAME
) -> str:
    """
    Calls Google's Gemini Flash model using the official google-genai SDK
    to generate grounded, truthful resume analysis and improvements.
    
    Strict constraints:
    - Never invents skills, experience, projects, certifications, or metrics.
    - Uses placeholders like '[add your actual metric if available]' when metrics are missing.
    - Clearly distinguishes existing qualifications from missing skills.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    from google import genai

    client = genai.Client(api_key=api_key)

    matching_skills_str = ", ".join(matching_skills) if matching_skills else "None"
    missing_skills_str = ", ".join(missing_skills) if missing_skills else "None"
    missing_kw_str = (
        ", ".join([f"{word} ({count}x in JD)" for word, count in missing_keywords[:8]])
        if missing_keywords
        else "None"
    )

    prompt = f"""You are an expert technical resume coach and career advisor.
You are evaluating a candidate's resume against a target job description.
You are also provided with deterministic keyword matching data from Version 1 of this tool.

### INPUT DATA:
1. DETERMINISTIC V1 METRICS:
   - Match Score: {match_score:.1f}%
   - Matching Skills in Resume: {matching_skills_str}
   - Missing Required Skills: {missing_skills_str}
   - Important Missing Keywords: {missing_kw_str}

2. CANDIDATE RESUME TEXT:
{resume_text}

3. TARGET JOB DESCRIPTION:
{jd_text}

---

### STRICT RULES & CONSTRAINTS (DO NOT VIOLATE):
- Base all evaluations and recommendations ONLY on the actual information present in the candidate's resume.
- Never invent skills.
- Never invent experience.
- Never invent projects.
- Never invent certifications.
- Never invent achievements.
- Never invent numerical metrics.
- Clearly distinguish existing skills from missing skills.
- Clearly distinguish future learning suggestions from current qualifications.
- If a metric would improve a bullet point but is unavailable in the resume text, write:
  [add your actual metric if available]

---

### REQUIRED RESPONSE STRUCTURE:
Format your response cleanly using the following markdown headings:

### 1. Overall Resume-to-Job Fit Explanation
Provide a clear, balanced explanation of how well the candidate's actual qualifications align with the job description, referencing the {match_score:.1f}% match score and key strengths.

### 2. Top 5 Improvement Priorities
Provide exactly 5 numbered, prioritized action items to strengthen this specific application. For each priority, include a concise explanation of WHY it matters.

### 3. Suggestions for Improving Existing Resume Sections
Give actionable advice for improving existing sections of the resume (e.g., Professional Summary, Technical Skills, Work Experience, Education/Projects) without fabricating new roles or qualifications.

### 4. Improved Experience & Project Bullet Points
Select 2 to 3 existing bullet points directly from the resume and rewrite them for stronger impact using action verbs and the Google X-Y-Z formula (Accomplished [X], as measured by [Y], by doing [Z]).
- Rely ONLY on facts and technologies already stated in the resume.
- If a metric is missing, use the placeholder '[add your actual metric if available]'.
- Provide a short reason for each revision explaining why it is more compelling.

### 5. Relevant Keywords & Terminology Alignment
Identify legitimate keywords from the job description that accurately describe the candidate's existing experience, and suggest where they can be naturally integrated without misrepresentation.
"""

    from google.genai import errors

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
    except errors.ServerError as server_err:
        # If the primary model experiences a temporary 503 spike, attempt fallback
        if "503" in str(server_err) and model_name != "gemini-3-flash-preview":
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
        else:
            raise server_err

    if not response or not response.text:
        raise ValueError("Received an empty response from Gemini.")

    return response.text.strip()


# ------------------------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume & Job Description Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Sidebar Information
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About the Analyzer")
    st.markdown(
        """
        **AI Resume Optimizer** combines two complementary layers of analysis:
        
        **Layer 1: Deterministic Engine (V1)**
        - Fast, 100% transparent keyword matching.
        - Checks 35+ industry-standard technical skills.
        - Calculates exact match percentage and keyword gaps.
        - Runs locally without requiring an API key.
        
        **Layer 2: AI-Powered Deep Analysis (V2)**
        - Contextual understanding using Google's **Gemini Flash**.
        - Rewrites bullet points using the Google X-Y-Z formula.
        - Suggests section-by-section improvements.
        - Strictly grounded in the resume (no hallucinations).
        
        ---
        🔒 **Security**: `GEMINI_API_KEY` is read locally from `.env` and is never exposed or logged.
        """
    )
    
    with st.expander("📚 Tracked Technical Skills (V1)"):
        st.write(", ".join(sorted(PREDEFINED_SKILLS.keys())))


# ------------------------------------------------------------------------------
# Main Header
# ------------------------------------------------------------------------------
st.title("📄 AI Resume & Job Description Analyzer")
st.caption("Version 2 • Deterministic Keyword Matching + Google Gemini AI Deep Analysis")

st.markdown(
    """
    Compare your resume with any target job description to uncover matching skills,
    missing technical requirements, and receive grounded AI-powered improvement suggestions.
    """
)

# Initialize Session State for Sample Job Description
if "job_description_input" not in st.session_state:
    st.session_state["job_description_input"] = ""


def set_sample_jd():
    st.session_state["job_description_input"] = SAMPLE_JOB_DESCRIPTION.strip()


# ------------------------------------------------------------------------------
# User Input Section (2 Columns)
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="medium")

# Column 1: Resume Upload
with col1:
    st.subheader("1. Candidate Resume")
    uploaded_pdf = st.file_uploader(
        "Upload Resume in PDF format",
        type=["pdf"],
        help="Upload a standard text-based PDF resume."
    )

    extracted_resume_text = ""
    if uploaded_pdf is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_resume_text = extract_text_from_pdf(uploaded_pdf)

        if extracted_resume_text.strip():
            word_count = len(extracted_resume_text.split())
            st.success(f"✅ Successfully extracted {word_count} words from PDF.")
            with st.expander("👁️ View Extracted Resume Text"):
                st.text_area(
                    "Raw Extracted Text",
                    extracted_resume_text,
                    height=200,
                    disabled=True,
                    label_visibility="collapsed"
                )
        else:
            st.error(
                "❌ Could not extract text from this PDF. It may be scanned, image-only, or password-protected. "
                "Please upload a standard text-based PDF."
            )

# Column 2: Job Description
with col2:
    st.subheader("2. Job Description")
    st.button(
        "📋 Load Sample Job Description",
        on_click=set_sample_jd,
        help="Click to automatically fill the text area with a realistic sample job description."
    )
    
    job_description_text = st.text_area(
        "Paste Job Description Here",
        value=st.session_state["job_description_input"],
        height=240,
        placeholder="Paste requirements, qualifications, and role responsibilities here...",
        help="Paste the full job posting text."
    )

st.markdown("---")

# ------------------------------------------------------------------------------
# Analysis Trigger & Validation
# ------------------------------------------------------------------------------
analyze_clicked = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)

if analyze_clicked:
    # Input Validation
    has_error = False

    if uploaded_pdf is None:
        st.warning("⚠️ Please upload a PDF resume before analyzing.")
        has_error = True
    elif not extracted_resume_text.strip():
        st.error("❌ The uploaded PDF contains no readable text. Please check the PDF file.")
        has_error = True

    if not job_description_text.strip():
        st.warning("⚠️ Please enter or paste a job description before analyzing.")
        has_error = True

    if not has_error:
        # Run Version 1 Analysis Pipeline (Always Executes First)
        with st.spinner("Analyzing resume against job description..."):
            results = analyze_resume_and_job(extracted_resume_text, job_description_text)

        # ======================================================================
        # VERSION 1: RESULTS DASHBOARD
        # ======================================================================
        st.subheader("📊 Match Overview (Version 1)")

        score = results["match_score"]
        matching_count = len(results["matching_skills"])
        missing_count = len(results["missing_skills"])
        required_count = results["required_skill_count"]
        resume_count = results["resume_skill_count"]

        # Display Metrics in Columns
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Match Score", f"{score:.1f}%")
        m_col2.metric("Matching Skills", matching_count)
        m_col3.metric("Missing Skills", missing_count)
        m_col4.metric("Job Skills Required", required_count)
        m_col5.metric("Resume Skills Found", resume_count)

        # Visual Progress Bar
        st.progress(score / 100.0)

        # Score Feedback Alert
        if required_count == 0:
            st.info(
                "ℹ️ No predefined technical skills were recognized in the job description. "
                "Check that you included the technical requirements section."
            )
        elif score >= 75.0:
            st.success(
                f"🌟 **Strong Technical Alignment ({score:.1f}%)**: Your resume covers most of the core technical "
                "requirements listed in this job description!"
            )
        elif score >= 50.0:
            st.info(
                f"💡 **Moderate Technical Alignment ({score:.1f}%)**: You match several key requirements, but bridging "
                "a few missing skills could significantly improve your fit."
            )
        else:
            st.warning(
                f"⚠️ **Low Technical Alignment ({score:.1f}%)**: Notable skill gaps were detected compared to the "
                "job description's technical requirements."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Skill Breakdown (Matching vs Missing)
        s_col1, s_col2 = st.columns(2)

        with s_col1:
            st.markdown("### ✅ Matching Skills")
            if results["matching_skills"]:
                st.success(f"**Found {matching_count} Matching Skill(s):**")
                skills_md = " ".join([f"`{skill}`" for skill in results["matching_skills"]])
                st.markdown(skills_md)
            else:
                st.warning("No overlapping technical skills detected.")

        with s_col2:
            st.markdown("### ❌ Missing Skills")
            if results["missing_skills"]:
                st.error(f"**Missing {missing_count} Required Skill(s):**")
                missing_md = " ".join([f"`{skill}`" for skill in results["missing_skills"]])
                st.markdown(missing_md)
            else:
                st.success("🎉 Great news! None of the tracked job description skills are missing from your resume.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Important Missing Keywords (Frequency Analysis)
        st.subheader("🔍 Important Keywords Missing from Resume")
        st.caption(
            "Identifies high-frequency terms appearing in the job description that were not found in your resume text."
        )

        if results["missing_keywords"]:
            kw_cols = st.columns(min(len(results["missing_keywords"]), 5))
            for idx, (word, count) in enumerate(results["missing_keywords"][:5]):
                with kw_cols[idx]:
                    st.metric(label=f"Keyword #{idx+1}", value=word, delta=f"{count}x in JD")

            with st.expander("📋 View All Missing Keywords with Frequency"):
                kw_data = [
                    {"Keyword": word, "Frequency in Job Description": f"{count} time(s)"}
                    for word, count in results["missing_keywords"]
                ]
                st.table(kw_data)
        else:
            st.info("No significant keyword gaps were detected between the job description and your resume.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Rule-Based Improvement Suggestions (V1)
        st.subheader("💡 Rule-Based Suggestions (Version 1)")
        st.caption("Heuristic recommendations based on your matching profile and keyword gaps.")

        for i, tip in enumerate(results["suggestions"], 1):
            if "Ethical" in tip:
                st.warning(f"**{i}.** {tip}")
            elif "Review Missing Skills" in tip or "Add Demonstrative Projects" in tip:
                st.info(f"**{i}.** {tip}")
            else:
                st.markdown(f"**{i}.** {tip}")

        # Expandable Details & Audit Section
        with st.expander("📂 Detailed Skill Breakdown & Cleaned Text (V1 Audit)"):
            det_col1, det_col2 = st.columns(2)
            with det_col1:
                st.markdown(f"**All Skills Detected in Resume ({len(results['resume_skills'])}):**")
                st.write(", ".join(results["resume_skills"]) if results["resume_skills"] else "None")
            with det_col2:
                st.markdown(f"**All Skills Detected in Job Description ({len(results['jd_skills'])}):**")
                st.write(", ".join(results["jd_skills"]) if results["jd_skills"] else "None")

            st.markdown("---")
            st.markdown("**Cleaned Text Preview (Used for Keyword Matching):**")
            prev_col1, prev_col2 = st.columns(2)
            with prev_col1:
                st.text_area("Preprocessed Resume", clean_text(extracted_resume_text), height=150, disabled=True)
            with prev_col2:
                st.text_area("Preprocessed Job Description", clean_text(job_description_text), height=150, disabled=True)

        # ======================================================================
        # VERSION 2: AI-POWERED RESUME ANALYSIS (GEMINI)
        # ======================================================================
        st.markdown("---")
        st.subheader("🤖 AI-Powered Resume Analysis (Version 2)")
        st.caption(f"Contextual evaluation powered by Google Gemini ({GEMINI_MODEL_NAME})")

        api_key = get_gemini_api_key()

        if not api_key:
            st.warning(
                "⚠️ **Gemini API Key Missing**: The AI-Powered Resume Analysis requires a Google Gemini API key. "
                "Add `GEMINI_API_KEY=your_key` to a local `.env` file in the project folder to enable this feature. "
                "The deterministic Version 1 analysis above remains fully functional."
            )
        else:
            with st.spinner(f"Generating deep AI analysis with Gemini ({GEMINI_MODEL_NAME})..."):
                try:
                    ai_feedback = generate_gemini_analysis(
                        resume_text=extracted_resume_text,
                        jd_text=job_description_text,
                        match_score=score,
                        matching_skills=results["matching_skills"],
                        missing_skills=results["missing_skills"],
                        missing_keywords=results["missing_keywords"],
                        model_name=GEMINI_MODEL_NAME
                    )
                    st.success("✅ Gemini AI Analysis Complete!")
                    st.markdown(ai_feedback)
                except ValueError as val_err:
                    st.warning(
                        f"⚠️ **Gemini Configuration Issue**: {str(val_err)}. "
                        "Your Version 1 keyword matching results above are completely unaffected."
                    )
                except Exception as err:
                    err_str = str(err).lower()
                    if "401" in err_str or "403" in err_str or "api_key_invalid" in err_str or "api key not valid" in err_str or "invalid api key" in err_str:
                        st.error(
                            "❌ **Invalid API Key**: The provided `GEMINI_API_KEY` was not accepted by the Gemini API. "
                            "Please check your `.env` file and verify that your key is active and correctly entered. "
                            "Your Version 1 keyword matching results above are completely unaffected."
                        )
                    elif "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
                        st.error(
                            "⏳ **API Quota / Rate Limit Exceeded**: You have temporarily exceeded your Gemini API rate limit or quota. "
                            "Please wait a few moments before trying again. "
                            "Your Version 1 keyword matching results above are completely unaffected."
                        )
                    elif "503" in err_str or "unavailable" in err_str or "high demand" in err_str or "overloaded" in err_str:
                        st.error(
                            "🌐 **Gemini Service Temporarily Busy**: Google's Gemini servers are currently experiencing high demand. "
                            "Please retry in a moment. "
                            "Your Version 1 keyword matching results above are completely unaffected."
                        )
                    else:
                        st.error(
                            "❌ **AI Analysis Unavailable**: An error occurred while communicating with the Gemini API. "
                            "Please verify your internet connection or API settings and try again. "
                            "Your Version 1 keyword matching results above are completely unaffected."
                        )

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>AI Resume & Job Description Analyzer • Version 2 (Hybrid Keyword Matching + Gemini Flash AI)</small></center>",
    unsafe_allow_html=True
)
