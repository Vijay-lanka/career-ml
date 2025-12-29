import pdfplumber
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# ---------------- CAREER ROLES & SKILLS ----------------

SKILLS = [
    "python", "java", "javascript", "react", "node",
    "sql", "machine learning", "data science",
    "next.js", "html", "css"
]

career_roles = {

    # ================= IT / SOFTWARE =================
    "Frontend Developer": ["javascript", "react", "css", "html", "next.js"],
    "Backend Developer": ["node.js", "python", "java", "sql", "api"],
    "Full Stack Developer": ["javascript", "react", "node.js", "sql", "next.js"],
    "Software Engineer": ["programming", "algorithms", "data structures", "oop"],
    "Web Developer": ["html", "css", "javascript", "frontend", "backend"],

    # ================= DATA & AI =================
    "Data Scientist": ["python", "sql", "machine learning", "statistics"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning"],
    "AI Engineer": ["python", "nlp", "deep learning", "ai"],
    "Data Analyst": ["sql", "excel", "python", "power bi", "tableau"],
    "Business Intelligence Analyst": ["sql", "power bi", "tableau", "dashboard"],
    "AI Researcher": ["python", "research", "deep learning", "nlp"],
    "Big Data Engineer": ["hadoop", "spark", "kafka", "big data"],

    # ================= CLOUD / DEVOPS =================
    "DevOps Engineer": ["docker", "kubernetes", "aws", "linux"],
    "Cloud Engineer": ["aws", "azure", "gcp"],
    "Site Reliability Engineer": ["monitoring", "linux", "cloud", "automation"],
    "System Administrator": ["linux", "networking", "server", "security"],

    # ================= MOBILE =================
    "Mobile App Developer": ["flutter", "react native", "android", "ios"],
    "Android Developer": ["java", "kotlin", "android"],
    "iOS Developer": ["swift", "ios", "xcode"],

    # ================= CYBER SECURITY =================
    "Cyber Security Analyst": ["security", "networking", "linux"],
    "Ethical Hacker": ["penetration testing", "security", "networking"],
    "SOC Analyst": ["security", "monitoring", "incident response"],

    # ================= CORE ENGINEERING =================
    "Mechanical Engineer": ["cad", "solidworks", "manufacturing", "design"],
    "Electrical Engineer": ["circuits", "power systems", "electronics"],
    "Civil Engineer": ["construction", "autocad", "surveying"],
    "Electronics Engineer": ["embedded systems", "microcontrollers", "iot"],
    "Robotics Engineer": ["robotics", "embedded", "python", "control systems"],

    # ================= BUSINESS / MANAGEMENT =================
    "Business Analyst": ["business analysis", "requirements", "documentation"],
    "Product Manager": ["product strategy", "roadmap", "agile"],
    "Project Manager": ["project management", "planning", "leadership"],
    "Operations Manager": ["operations", "process management", "optimization"],
    "Management Consultant": ["strategy", "problem solving", "analysis"],

    # ================= FINANCE =================
    "Financial Analyst": ["finance", "excel", "financial modeling"],
    "Investment Analyst": ["investments", "markets", "portfolio"],
    "Accountant": ["accounting", "taxation", "finance"],
    "Auditor": ["auditing", "compliance", "finance"],
    "Banking Professional": ["banking", "finance", "customer service"],

    # ================= MARKETING =================
    "Digital Marketing Specialist": ["seo", "content marketing", "google ads"],
    "Marketing Analyst": ["marketing", "analytics", "campaigns"],
    "Social Media Manager": ["social media", "branding", "content"],

    # ================= DESIGN =================
    "UI/UX Designer": ["ui design", "ux research", "figma", "prototyping"],
    "Graphic Designer": ["photoshop", "illustrator", "design"],
    "Product Designer": ["design thinking", "prototyping", "ui"],

    # ================= HEALTHCARE =================
    "Healthcare Data Analyst": ["healthcare", "data analysis", "statistics"],
    "Medical Coder": ["medical coding", "icd", "health records"],
    "Clinical Research Analyst": ["clinical research", "data analysis"],

    # ================= EDUCATION =================
    "Professor": ["teaching", "research", "subject knowledge"],
    "Lecturer": ["education", "teaching", "curriculum"],
    "Online Course Instructor": ["online teaching", "content creation"],

    # ================= GOVERNMENT / PUBLIC =================
    "Government Officer": ["administration", "policy", "public service"],
    "Public Sector Analyst": ["policy analysis", "research"],
    "Defense Services": ["leadership", "discipline", "training"],

    # ================= MISC =================
    "Technical Writer": ["documentation", "writing", "technology"],
    "Quality Assurance Engineer": ["testing", "automation", "qa"],
    "Supply Chain Analyst": ["logistics", "supply chain", "operations"],
    "Entrepreneur": ["business", "startup", "innovation"],
}

all_skills = sorted(
    set(skill for skills_list in career_roles.values() for skill in skills_list)
)

# ---------------- FUNCTIONS ----------------

def extract_text_from_pdf(pdf_path_or_bytes):
    text = ""
    with pdfplumber.open(pdf_path_or_bytes) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def extract_skills(text):
    text = text.lower()
    return list(set(skill for skill in SKILLS if skill in text))

def resume_features(text, skills):
    sections = ["education", "skills", "experience", "projects", "internship"]
    return {
        "word_count": len(text.split()),
        "skill_count": len(skills),
        "section_score": sum(1 for sec in sections if sec in text.lower())
    }

def calculate_resume_score(features):
    score = 0
    if features["word_count"] >= 400:
        score += 30
    elif features["word_count"] >= 250:
        score += 20
    else:
        score += 10
    score += min(features["skill_count"] * 5, 40)
    score += features["section_score"] * 7
    return min(score, 100)

def vectorize_skills(input_skills):
    return [1 if skill in input_skills else 0 for skill in all_skills]

# ---------------- MAIN FUNCTION ----------------

def analyze_resume(pdf_path_or_bytes):
    text = extract_text_from_pdf(pdf_path_or_bytes)
    skills = extract_skills(text)
    features = resume_features(text, skills)
    score = calculate_resume_score(features)

    # ML predictions
    resume_vector = vectorize_skills(skills)

    data = []
    labels = []
    for role, role_skills in career_roles.items():
        data.append(vectorize_skills(role_skills))
        labels.append(role)

    X = pd.DataFrame(data, columns=all_skills)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X, y_encoded)

    resume_df = pd.DataFrame([resume_vector], columns=all_skills)
    probs = model.predict_proba(resume_df)[0]
    top_indices = probs.argsort()[-3:][::-1]
    top_roles = label_encoder.inverse_transform(top_indices)

    top_predictions = [
        {"career": top_roles[i], "confidence": float(probs[top_indices[i]])}
        for i in range(len(top_roles))
    ]

    return {
        "file_name": getattr(pdf_path_or_bytes, "name", "Uploaded Resume"),
        "score": score,
        "skills": skills,
        "insights": [
            f"Word count: {features['word_count']}",
            f"Number of skills detected: {features['skill_count']}",
            f"Sections found: {features['section_score']}"
        ],
        "career_primary": top_roles[0],
        "career_alternatives": list(top_roles[1:]),
        "predictions": top_predictions
    }
