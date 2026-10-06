# 📄 AI Resume & Job Description Analyzer (Version 2)

A beginner-friendly, fast, and transparent web application that compares a candidate's resume (PDF) with a target job description. It provides both deterministic keyword matching (Version 1) and grounded contextual AI analysis using Google's Gemini Flash model (Version 2).

> ℹ️ **Architecture Overview:**  
> This project is designed in two complementary, independent layers:
> 1. **Version 1 (Deterministic Engine):** 100% local, transparent keyword-based skill extraction and frequency analysis.
> 2. **Version 2 (Gemini Integration):** Grounded contextual resume coaching powered by Google's official `google-genai` SDK (`gemini-3.5-flash-lite`).
>
> It does **NOT** use LangChain, LangGraph, RAG, vector databases, or autonomous AI agents.

---

## 📌 Problem Statement

Job seekers often struggle to gauge how closely their resumes match the specific qualifications and terminology requested in job descriptions. 
- **Keyword gaps**: Resumes often omit technical tools or terminology that the candidate actually knows.
- **Generic bullet points**: Experience bullet points frequently list generic responsibilities rather than impactful achievements.
- **ATS opacity**: Applicants do not know which missing skills create the largest alignment gap.

The **AI Resume & Job Description Analyzer** addresses these challenges by offering fast, objective feedback on technical alignment, accompanied by actionable recommendations grounded strictly in the candidate's actual background.

---

## ✨ Features

### Version 1 (Deterministic Core)
- **PDF Resume Extraction**: Reads digital text from PDF resumes using `pypdf`.
- **Predefined Skill Dictionary**: Automatically detects 35+ industry-standard technical skills (languages, frameworks, cloud, databases).
- **Word-Boundary Matching**: Exact regex patterns prevent false positives (e.g., `Java` does not falsely match `JavaScript`, and `Git` does not match `GitHub`).
- **Deterministic Match Scoring**: Percentage score based on the proportion of required job skills present on the resume.
- **Missing Skills Identification**: Instantly surfaces required technologies missing from the resume.
- **Keyword Frequency Analysis**: Discovers non-stopword keywords appearing frequently in the job description that do not appear in the resume.
- **Rule-Based Guidance**: Practical tips for tailoring resumes honestly without exaggerating experience.
- **Standalone CLI**: A terminal interface (`main.py`) that operates completely independently without requiring Gemini or an API key.

### Version 2 (Gemini Flash Integration)
- **AI-Powered Contextual Analysis**: Leverages Google's official `google-genai` Python SDK.
- **Supported Flash Model**: Configured to use a currently supported Gemini Flash model (`gemini-3.5-flash-lite`), stored in a single configuration constant.
- **Grounding & Honesty Enforcement**: Strictly instructed to evaluate only information present in the resume. Never fabricates skills, achievements, or metrics.
- **Metric Placeholders**: Uses `[add your actual metric if available]` when quantitative metrics are missing.
- **Actionable AI Feedback**:
  1. Overall resume-to-job fit explanation.
  2. Top 5 prioritized improvement actions.
  3. Suggestions for improving existing resume sections.
  4. 2–3 rewritten experience/project bullet points using the Google X-Y-Z formula.
  5. Relevant keywords from the job description that accurately describe existing experience.
- **Resilient Fallback**: If `GEMINI_API_KEY` is missing or the Gemini API fails, Version 1 analysis is displayed completely unaffected.

---

## ⚙️ How It Works

```
Candidate Resume (PDF)                 Target Job Description
          │                                      │
          ▼                                      ▼
   pypdf Extraction                      Raw Text Input
          │                                      │
          └──────────────────┬───────────────────┘
                             │
                             ▼
                  Text Preprocessing
              (lower, strip symbols, norm)
                             │
                             ▼
              Deterministic Skill Matching (V1)
            (35 Predefined Skills + Regex Patterns)
                             │
                             ├───────────────┐
                             ▼               ▼
                      Matching Skills   Missing Skills
                             │               │
                             └───────┬───────┘
                                     │
                                     ▼
                                Match Score
               (matching JD skills / unique JD skills * 100)
                                     │
                                     ▼
                         Missing Keyword Frequency
                        (top non-stopwords in JD)
                                     │
                                     ▼
                     V1 Dashboard & Rule-Based Tips
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │                                                   │
  (No API Key configured)                         (GEMINI_API_KEY present)
           │                                                   │
           ▼                                                   ▼
Displays clear warning;                          Google GenAI SDK (V2)
V1 results remain 100% usable                    Calls gemini-3.5-flash-lite
                                                 Grounded deep analysis &
                                                 rewritten bullet points
```

---

## 🔐 API-Key Security & Configuration

Version 2 reads your Google Gemini API key securely from a local `.env` file using `python-dotenv`:

1. Create a `.env` file in the project root directory:
   ```env
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
2. **Never hardcode, log, or display the API key**: The code only accesses `os.environ.get("GEMINI_API_KEY")` and never prints or reveals the key in terminal logs, Streamlit UI, or exception traces.
3. **Git Protection**: `.env` and `.streamlit/secrets.toml` are strictly included in `.gitignore` to prevent accidental commits.

---

## 📊 Match Score Methodology (V1)

The Version 1 match score is calculated with a deterministic, division-by-zero-safe formula:

$$\text{Match Score} = \left( \frac{\text{Number of Job Description Skills Found in Resume}}{\text{Total Unique Recognized Skills in Job Description}} \right) \times 100$$

If the job description contains no recognized skills from the 35-skill dictionary, the system safely reports `0.0%` with a clarifying message rather than raising an arithmetic error.

---

## 🛑 Limitations

### Version 1 Limitations:
- **Keyword-Only Matching**: Cannot understand semantic equivalents (e.g., will not equate "Relational Databases" with "PostgreSQL" unless the exact keyword is present).
- **Scanned / Flat Image PDFs**: `pypdf` extracts digital text only; scanned image PDFs require an OCR tool.
- **Fixed Skill Scope**: Technical tools outside the 35 predefined skills are captured only through keyword frequency analysis.

### Version 2 Limitations:
- **Requires Internet & API Quota**: AI analysis relies on connectivity to Google's Gemini API and requires a valid API key with available quota.
- **Grounded Scope**: Gemini is strictly constrained to the text provided in the resume. It will not make assumptions or fill in missing professional background.

---

## 💻 Technologies Used

- **Python 3.10+** (tested on Python 3.12)
- **Streamlit**: Web application dashboard
- **pypdf**: PDF text extraction
- **google-genai**: Official Google GenAI Python SDK
- **python-dotenv**: Local environment variable management

---

## 📁 Project Structure

```
ai-resume-optimizer/
│
├── app.py              # Streamlit web app (V1 Engine + V2 Gemini Analysis)
├── main.py             # Pure Python V1 engine & CLI (Independent of Gemini)
├── sample_resume.pdf   # Sample resume PDF for testing
├── requirements.txt    # Clean dependencies (streamlit, pypdf, google-genai, python-dotenv)
├── README.md           # Documentation
├── .gitignore          # Ignores .env, .streamlit/secrets.toml, __pycache__, etc.
└── .env                # Local secrets file (ignored by Git, never committed)
```

---

## 🚀 Installation & Setup

### 1. Clone or Open the Repository
```bash
cd ai-resume-optimizer
```

### 2. (Optional) Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### Option A: Run the Streamlit Web Application (V1 + V2)

Launch the web UI:
```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py` if Streamlit is in your system PATH)*

1. Open `http://localhost:8501`.
2. Upload a PDF resume (you can test with the provided `sample_resume.pdf`).
3. Paste a job description or click **📋 Load Sample Job Description**.
4. Click **🚀 Analyze Resume**:
   - **Running WITH Gemini**: If `GEMINI_API_KEY` is present in `.env`, the app renders the V1 score and metrics, followed immediately by the AI-powered deep analysis and rewritten bullet points.
   - **Running WITHOUT Gemini**: If no key is configured, the app renders all V1 results and displays an informative warning in the AI section explaining how to add a key.

---

### Option B: Run via Command Line Interface (CLI - V1 Core)

`main.py` is completely independent of Gemini and requires no API key:

#### 1. Demo Mode (with built-in sample data):
```bash
python main.py
```

#### 2. Analyze with a PDF Resume and Custom Job Description:
```bash
python main.py --resume sample_resume.pdf --jd "Looking for Python, Docker, AWS, FastAPI, PyTorch, MongoDB developer."
```

#### 3. View CLI Options:
```bash
python main.py --help
```
