"""
AI Resume & Job Description Analyzer - Version 1 (Core & CLI)
============================================================
A beginner-friendly, keyword-based resume and job description analysis engine.
Version 1 relies on transparent rule-based and frequency matching (no LLMs or vector databases).
"""

import os
import re
import sys
import argparse
from collections import Counter
import pypdf

# ==============================================================================
# 1. PREDEFINED SKILL DICTIONARY
# ==============================================================================
# Practical dictionary of common skills with regex patterns for accurate matching.
# Boundary patterns ensure skills like 'Java' do not falsely match 'JavaScript',
# and 'Git' does not match 'GitHub'.
PREDEFINED_SKILLS = {
    "Python": [r"\bpython\b"],
    "C++": [r"\bc\+\+(?=[^\w]|$)"],
    "Java": [r"\bjava\b"],
    "JavaScript": [r"\bjavascript\b"],
    "SQL": [r"\bsql\b"],
    "HTML": [r"\bhtml5?\b"],
    "CSS": [r"\bcss3?\b"],
    "Git": [r"\bgit\b"],
    "GitHub": [r"\bgithub\b"],
    "Machine Learning": [r"\bmachine[\s-]learning\b"],
    "Deep Learning": [r"\bdeep[\s-]learning\b"],
    "NLP": [r"\bnlp\b", r"\bnatural[\s-]language[\s-]processing\b"],
    "Generative AI": [r"\bgenerative[\s-]ai\b", r"\bgen[\s-]?ai\b"],
    "LangChain": [r"\blangchain\b"],
    "LangGraph": [r"\blanggraph\b"],
    "RAG": [r"\brag\b", r"\bretrieval[\s-]augmented[\s-]generation\b"],
    "TensorFlow": [r"\btensorflow\b", r"\btf\b"],
    "PyTorch": [r"\bpytorch\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Scikit-learn": [r"\bscikit[\s-]learn\b", r"\bsklearn\b"],
    "Matplotlib": [r"\bmatplotlib\b"],
    "Seaborn": [r"\bseaborn\b"],
    "Docker": [r"\bdocker\b"],
    "AWS": [r"\baws\b", r"\bamazon[\s-]web[\s-]services\b"],
    "Azure": [r"\bazure\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle[\s-]cloud\b"],
    "React": [r"\breact(?:\.js)?\b"],
    "Node.js": [r"\bnode(?:\.js)?\b", r"\bnodejs\b"],
    "Flask": [r"\bflask\b"],
    "FastAPI": [r"\bfastapi\b", r"\bfast[\s-]api\b"],
    "Streamlit": [r"\bstreamlit\b"],
    "MongoDB": [r"\bmongodb\b", r"\bmongo\b"],
    "MySQL": [r"\bmysql\b"],
    "PostgreSQL": [r"\bpostgresql\b", r"\bpostgres\b"]
}

# Common English stop words to exclude during keyword frequency analysis
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more",
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
    "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you",
    "your", "yours", "yourself", "yourselves", "will", "shall", "must", "may", "might", "also",
    "including", "across", "well", "using", "work", "working", "looking", "candidate", "role",
    "years", "experience", "required", "preferred", "join", "team", "responsibilities",
    "qualifications", "strong", "ability", "plus", "knowledge", "skills", "proven", "etc"
}


# ==============================================================================
# 2. RESUME PDF EXTRACTION
# ==============================================================================
def extract_text_from_pdf(pdf_source) -> str:
    """
    Extracts readable text from a PDF file using pypdf.
    
    Args:
        pdf_source: A file path string or a file-like object (e.g. UploadedFile).
        
    Returns:
        Extracted text as a string, or an empty string if unreadable/empty.
    """
    if pdf_source is None:
        return ""
        
    try:
        # Reset stream position if it's a file-like object
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
            
        reader = pypdf.PdfReader(pdf_source)
        if len(reader.pages) == 0:
            return ""

        extracted_text_list = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text_list.append(page_text)

        full_text = "\n".join(extracted_text_list).strip()
        return full_text
    except Exception as error:
        # Gracefully handle corrupted, encrypted, or invalid PDFs
        print(f"[Warning] Failed to extract text from PDF: {error}")
        return ""


# ==============================================================================
# 3. TEXT PREPROCESSING
# ==============================================================================
def clean_text(text: str) -> str:
    """
    Preprocesses text for keyword matching:
    - Converts text to lowercase
    - Normalizes whitespace
    - Removes punctuation while preserving symbols necessary for technical terms (+, ., -)
    
    Args:
        text: Raw text string.
        
    Returns:
        Cleaned, lowercased text string.
    """
    if not text:
        return ""

    # Convert to lowercase
    cleaned = text.lower()

    # Replace newlines and tabs with spaces
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)

    # Remove non-alphanumeric punctuation except +, ., -, and #
    cleaned = re.sub(r'[()[\]{}"\':;,!?~@$%^*\\/|<>]+', " ", cleaned)

    # Remove trailing periods at word or sentence boundaries, preserving internal dots like node.js
    cleaned = re.sub(r"\.(?=\s|$)", " ", cleaned)

    # Collapse multiple spaces into a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ==============================================================================
# 4. SKILL EXTRACTION
# ==============================================================================
def extract_skills(text: str) -> list[str]:
    """
    Identifies which predefined skills appear in the given text using keyword matching.
    
    Args:
        text: The input text (resume or job description).
        
    Returns:
        A sorted list of unique skill names identified in the text.
    """
    if not text:
        return []

    cleaned = clean_text(text)
    detected_skills = []

    for skill_name, patterns in PREDEFINED_SKILLS.items():
        for pattern in patterns:
            if re.search(pattern, cleaned):
                detected_skills.append(skill_name)
                break

    return sorted(detected_skills)


# ==============================================================================
# 5. MATCHING ANALYSIS
# ==============================================================================
def calculate_match(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Calculates the matching metrics between resume skills and job description skills.
    
    Formula:
        Match Score = (number of job-description skills found in resume /
                       number of unique job-description skills) * 100
                       
    Args:
        resume_skills: List of skills detected in the resume.
        jd_skills: List of skills detected in the job description.
        
    Returns:
        Dictionary containing matching_skills, missing_skills, counts, and match_score.
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matching_skills = sorted(list(resume_set & jd_set))
    missing_skills = sorted(list(jd_set - resume_set))

    resume_skill_count = len(resume_set)
    required_skill_count = len(jd_set)

    # Gracefully prevent division by zero if the job description contains no recognized skills
    if required_skill_count > 0:
        match_score = round((len(matching_skills) / required_skill_count) * 100, 1)
    else:
        match_score = 0.0

    return {
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "resume_skill_count": resume_skill_count,
        "required_skill_count": required_skill_count,
        "match_score": match_score
    }


# ==============================================================================
# 6. EXPERIENCE / KEYWORD FREQUENCY ANALYSIS
# ==============================================================================
def analyze_keywords(resume_text: str, jd_text: str, detected_skills: list[str], top_n: int = 10) -> list[tuple[str, int]]:
    """
    Identifies high-frequency keywords from the job description that are absent
    from the resume using transparent frequency counting.
    
    Args:
        resume_text: Raw or cleaned text of the candidate's resume.
        jd_text: Raw or cleaned text of the job description.
        detected_skills: Predefined skills already detected (to avoid duplicates).
        top_n: Maximum number of missing keywords to return.
        
    Returns:
        List of tuples: (missing_keyword, occurrence_count_in_jd).
    """
    if not jd_text or not resume_text:
        return []

    # Extract all lowercase words with 3 or more characters
    resume_words = set(re.findall(r"\b[a-z]{3,}\b", resume_text.lower()))
    jd_words = re.findall(r"\b[a-z]{3,}\b", jd_text.lower())

    # Build set of words already counted as recognized technical skills
    skill_words = set()
    for skill in detected_skills:
        for word in re.findall(r"\b[a-z]{3,}\b", skill.lower()):
            skill_words.add(word)

    # Filter out stopwords, recognized skills, and pure numeric tokens
    meaningful_jd_words = [
        word for word in jd_words
        if word not in STOPWORDS and word not in skill_words and not word.isdigit()
    ]

    word_counts = Counter(meaningful_jd_words)

    missing_keywords = []
    for word, count in word_counts.most_common():
        if word not in resume_words:
            missing_keywords.append((word, count))
            if len(missing_keywords) >= top_n:
                break

    return missing_keywords


# ==============================================================================
# 7. RESUME IMPROVEMENT SUGGESTIONS
# ==============================================================================
def generate_suggestions(match_data: dict, missing_keywords: list[tuple[str, int]], resume_text: str) -> list[str]:
    """
    Generates rule-based resume improvement recommendations.
    Does NOT invent fake qualifications or experiences.
    
    Args:
        match_data: Results dictionary from calculate_match().
        missing_keywords: List of missing keywords and counts.
        resume_text: The candidate's resume text.
        
    Returns:
        List of practical, actionable suggestion strings.
    """
    suggestions = []
    missing_skills = match_data.get("missing_skills", [])
    match_score = match_data.get("match_score", 0.0)
    required_count = match_data.get("required_skill_count", 0)

    # 1. Missing technical skills guidance
    if missing_skills:
        sample_skills = ", ".join(missing_skills[:5])
        suffix = "..." if len(missing_skills) > 5 else ""
        suggestions.append(
            f"Review Missing Skills: The job description explicitly requires [{sample_skills}{suffix}]. "
            "If you possess hands-on experience or coursework with any of these tools, ensure they are explicitly listed in your skills section."
        )

    # 2. Project demonstration recommendation
    if missing_skills:
        top_skills = ", ".join(missing_skills[:3])
        suggestions.append(
            f"Add Demonstrative Projects: For key required tools such as {top_skills}, consider adding 1-2 portfolio projects "
            "or linking to GitHub repositories where you have applied them."
        )

    # 3. Measurable outcomes / impact checking
    has_metrics = bool(re.search(r"\d+[%xXkK]?|\$\d+", resume_text))
    if not has_metrics:
        suggestions.append(
            "Quantify Your Impact: Include measurable metrics (e.g., 'improved throughput by 25%', 'reduced API latency by 120ms', "
            "'supported 10,000+ active users') to substantiate your achievements."
        )
    else:
        suggestions.append(
            "Maintain Metric-Driven Bullet Points: You already include numerical achievements. Continue highlighting specific results "
            "and performance gains associated with your projects."
        )

    # 4. Terminology alignment with the job description
    if missing_keywords:
        top_kws = ", ".join([kw[0] for kw in missing_keywords[:5]])
        suggestions.append(
            f"Align Terminology: The job posting frequently uses keywords like [{top_kws}]. "
            "If this terminology matches your past responsibilities, align your bullet points with these industry-standard terms."
        )

    # 5. Score-based overall feedback
    if required_count == 0:
        suggestions.append(
            "Job Description Clarification: No recognized technical skills were found in the job description. "
            "Make sure you included the full requirements or technical qualifications section."
        )
    elif match_score >= 80:
        suggestions.append(
            "High Skill Alignment: Your technical profile is well-aligned with this role! Focus your efforts on behavioral preparation "
            "and deep architectural explanations of your listed projects."
        )
    elif match_score >= 50:
        suggestions.append(
            "Moderate Skill Alignment: Your background covers several core areas. Bridging the gap on 2-3 prominent missing skills "
            "can significantly strengthen your application."
        )
    else:
        suggestions.append(
            "Skill Gap Identified: There is a notable gap between your listed skills and this job's requirements. "
            "Target roles matching your current stack, or build targeted projects to demonstrate proficiency before applying."
        )

    # 6. Integrity and authenticity reminder
    suggestions.append(
        "Ethical Reminder: Do NOT add skills, tools, or experiences that you do not genuinely understand. "
        "Hiring managers assess technical proficiency during coding interviews."
    )

    return suggestions


# ==============================================================================
# 8. COMPLETE PIPELINE ANALYZER
# ==============================================================================
def analyze_resume_and_job(resume_text: str, jd_text: str) -> dict:
    """
    Executes the full Version 1 analysis pipeline given resume text and job description text.
    
    Args:
        resume_text: Extracted text from candidate's resume.
        jd_text: Job description text.
        
    Returns:
        Structured dictionary with all metrics, skill sets, keywords, and suggestions.
    """
    # Extract skills
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    # Calculate match statistics
    match_data = calculate_match(resume_skills, jd_skills)

    # Run keyword frequency analysis
    all_detected_skills = list(set(resume_skills + jd_skills))
    missing_keywords = analyze_keywords(resume_text, jd_text, all_detected_skills, top_n=10)

    # Generate rule-based recommendations
    suggestions = generate_suggestions(match_data, missing_keywords, resume_text)

    return {
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
        "matching_skills": match_data["matching_skills"],
        "missing_skills": match_data["missing_skills"],
        "resume_skill_count": match_data["resume_skill_count"],
        "required_skill_count": match_data["required_skill_count"],
        "match_score": match_data["match_score"],
        "missing_keywords": missing_keywords,
        "suggestions": suggestions
    }


# ==============================================================================
# 9. COMMAND LINE INTERFACE (CLI)
# ==============================================================================
# Sample data for demonstration mode
SAMPLE_RESUME = """
Jane Doe
Email: jane.doe@example.com | GitHub: github.com/janedoe | LinkedIn: linkedin.com/in/janedoe

PROFESSIONAL SUMMARY:
Backend and Machine Learning Engineer with 3+ years of experience designing scalable APIs,
microservices, and data pipelines. Proficient in Python, SQL, Git, and Docker.

TECHNICAL SKILLS:
- Languages: Python, SQL, C++, HTML, CSS
- Frameworks & Libraries: FastAPI, Flask, Pandas, NumPy, Scikit-learn, PyTorch
- Tools & Platforms: Git, GitHub, Docker, AWS (S3, EC2), PostgreSQL, MySQL
- Concepts: REST APIs, Machine Learning, Data Preprocessing, Unit Testing

WORK EXPERIENCE:
Software Engineer | DataTech Solutions (2022 - Present)
- Developed and maintained 15+ RESTful APIs using FastAPI and PostgreSQL, serving 50,000+ daily requests.
- Integrated PyTorch models into backend services, reducing inference time by 28%.
- Containerized applications using Docker and deployed cloud services to AWS.
- Collaborated across agile development teams using Git and GitHub for version control.

Junior Developer | CloudSphere (2021 - 2022)
- Built automated data cleaning pipelines using Python and Pandas.
- Designed database schemas and optimized complex SQL queries for PostgreSQL.
"""

SAMPLE_JOB_DESCRIPTION = """
Job Title: Senior AI & Backend Developer
Company: NextGen AI Technologies

We are seeking a talented Senior AI & Backend Developer to join our core engineering team.
You will architect high-performance APIs, implement Generative AI and RAG workflows,
and scale cloud infrastructure.

RESPONSIBILITIES:
- Architect, build, and deploy production microservices with Python, FastAPI, and Docker.
- Develop cutting-edge LLM and RAG pipelines using LangChain, LangGraph, and PyTorch.
- Design database architectures using PostgreSQL and MongoDB.
- Build CI/CD deployment pipelines on AWS and GCP.
- Collaborate closely with product managers and engineers in an Agile sprint environment.
- Conduct code reviews, mentor junior engineers, and uphold rigorous software quality standards.

QUALIFICATIONS & SKILLS:
- 3+ years of professional backend software development experience.
- Strong proficiency in Python, FastAPI, Docker, and Git.
- Practical experience with Machine Learning, Generative AI, RAG, and LangChain.
- Working knowledge of GCP, AWS, and PostgreSQL.
- Experience with LangGraph is a plus.
"""


def print_cli_report(results: dict):
    """Prints a clean, formatted report of analysis results to stdout."""
    print("\n" + "=" * 70)
    print("      AI RESUME & JOB DESCRIPTION ANALYZER (VERSION 1) REPORT      ")
    print("=" * 70)
    print("NOTE: Version 1 uses keyword-based matching (no LLMs or embeddings).\n")

    # Match Score & Summary
    score = results["match_score"]
    req_count = results["required_skill_count"]
    match_count = len(results["matching_skills"])
    print(f"[*] MATCH SCORE: {score:.1f}%")
    print(f"   - Job Description Skills Found: {req_count}")
    print(f"   - Matching Skills in Resume:    {match_count}")
    print(f"   - Missing Skills in Resume:     {len(results['missing_skills'])}")
    print(f"   - Total Resume Skills Found:    {results['resume_skill_count']}")
    print("-" * 70)

    # Matching Skills
    print(f"\n[+] MATCHING SKILLS ({len(results['matching_skills'])}):")
    if results["matching_skills"]:
        print("   " + ", ".join(results["matching_skills"]))
    else:
        print("   None detected.")

    # Missing Skills
    print(f"\n[-] MISSING SKILLS ({len(results['missing_skills'])}):")
    if results["missing_skills"]:
        print("   " + ", ".join(results["missing_skills"]))
    else:
        print("   None! All required technical skills appear on your resume.")

    # All Skills Overview
    print("\n[i] SKILL BREAKDOWN:")
    print("   Resume Skills: " + (", ".join(results["resume_skills"]) if results["resume_skills"] else "None"))
    print("   Job Skills:    " + (", ".join(results["jd_skills"]) if results["jd_skills"] else "None"))

    # Missing Keywords from JD
    print("\n[?] IMPORTANT MISSING KEYWORDS (Job Description Frequency):")
    if results["missing_keywords"]:
        for word, count in results["missing_keywords"]:
            print(f"   * {word} (occurs {count}x in JD)")
    else:
        print("   No significant missing keywords detected.")

    # Rule-Based Suggestions
    print("\n[!] RESUME IMPROVEMENT SUGGESTIONS:")
    for i, suggestion in enumerate(results["suggestions"], 1):
        clean_sugg = suggestion.replace("**", "").replace("`", "'")
        print(f"   {i}. {clean_sugg}")

    print("\n" + "=" * 70 + "\n")


def main():
    """CLI Entry Point."""
    parser = argparse.ArgumentParser(
        description="Version 1 AI Resume & Job Description Analyzer (Keyword-Based)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to resume PDF file or text file"
    )
    parser.add_argument(
        "--jd",
        type=str,
        help="Path to job description text file or raw job description string"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run analysis using realistic built-in sample data"
    )

    args = parser.parse_args()

    # Determine resume text
    resume_text = ""
    jd_text = ""

    if args.sample or (not args.resume and not args.jd):
        print("\n[Info] Running analysis in DEMO MODE with realistic sample data...")
        resume_text = SAMPLE_RESUME
        jd_text = SAMPLE_JOB_DESCRIPTION
    else:
        # Load resume
        if args.resume:
            if not os.path.exists(args.resume):
                print(f"[Error] Resume file not found at: {args.resume}")
                sys.exit(1)
            if args.resume.lower().endswith(".pdf"):
                print(f"[Info] Extracting text from PDF resume: {args.resume}")
                resume_text = extract_text_from_pdf(args.resume)
            else:
                with open(args.resume, "r", encoding="utf-8", errors="ignore") as f:
                    resume_text = f.read()

        # Load job description
        if args.jd:
            if os.path.exists(args.jd):
                with open(args.jd, "r", encoding="utf-8", errors="ignore") as f:
                    jd_text = f.read()
            else:
                jd_text = args.jd

    if not resume_text.strip():
        print("[Error] Resume text is empty or could not be extracted.")
        sys.exit(1)

    if not jd_text.strip():
        print("[Error] Job description is empty.")
        sys.exit(1)

    results = analyze_resume_and_job(resume_text, jd_text)
    print_cli_report(results)


if __name__ == "__main__":
    main()
