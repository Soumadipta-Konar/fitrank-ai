import numpy as np


IMPORTANT_SKILLS = {
    "python": 1.0,
    "machine learning": 1.2,
    "ml": 0.7,
    "deep learning": 0.8,
    "pytorch": 1.0,
    "transformers": 1.0,
    "nlp": 1.5,
    "llm": 1.2,
    "rag": 1.0,
    "fine-tuning": 1.1,
    "finetuning": 1.1,
    "lora": 0.9,
    "qlora": 0.9,
    "embeddings": 1.8,
    "embedding": 1.8,
    "sentence-transformers": 1.7,
    "semantic search": 1.8,
    "hybrid search": 1.7,
    "retrieval": 1.8,
    "information retrieval": 1.9,
    "ranking": 1.9,
    "recommendation": 1.5,
    "recommender": 1.5,
    "search": 1.3,
    "vector search": 1.7,
    "faiss": 1.7,
    "milvus": 1.5,
    "pinecone": 1.5,
    "qdrant": 1.5,
    "weaviate": 1.5,
    "elasticsearch": 1.3,
    "opensearch": 1.3,
    "ndcg": 1.5,
    "mrr": 1.4,
    "map": 1.1,
    "a/b testing": 1.2,
    "ab testing": 1.2,
}

PRODUCTION_TERMS = [
    "production", "deployed", "shipped", "launched", "real users",
    "latency", "monitoring", "scale", "large-scale", "pipeline",
    "api", "backend", "system", "systems", "owned", "end-to-end",
    "inference", "evaluation", "a/b", "experiment", "offline", "online",
]

BEST_TITLES = [
    "ai engineer", "ml engineer", "machine learning engineer",
    "applied scientist", "nlp engineer", "search engineer",
    "ranking engineer", "recommendation engineer", "recommender engineer",
]

GOOD_TITLES = [
    "software engineer", "backend engineer", "data scientist",
    "data engineer", "analytics engineer", "platform engineer",
    "research engineer",
]

BAD_TITLES = [
    "recruiter", "hr", "talent acquisition", "sales", "marketing",
    "designer", "graphic", "content writer", "finance", "accountant",
    "operations manager", "customer support",
]

CONSULTING_COMPANIES = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "mindtree", "ltimindtree", "tech mahindra", "hcl",
]

PREFERRED_LOCATIONS = [
    "pune", "noida", "delhi", "gurgaon", "gurugram", "ncr",
    "mumbai", "hyderabad", "bangalore", "bengaluru",
]

NEGATIVE_DOMAINS = [
    "photoshop", "graphic design", "speech recognition", "tts",
    "robotics", "computer vision", "image classification",
]


def contains_any(text, terms):
    text = str(text).lower()
    return any(term in text for term in terms)


def count_any(text, terms):
    text = str(text).lower()
    return sum(1 for term in terms if term in text)


def title_score(title):
    title = str(title).lower()

    if contains_any(title, BAD_TITLES):
        return 0.03

    if contains_any(title, BEST_TITLES):
        return 1.0

    if contains_any(title, GOOD_TITLES):
        return 0.68

    return 0.25


def experience_score(years):
    years = float(years or 0)

    if 5 <= years <= 9:
        return 1.0
    if 4 <= years < 5:
        return 0.78
    if 9 < years <= 11:
        return 0.72
    if 3 <= years < 4:
        return 0.48
    if 11 < years <= 14:
        return 0.42

    return 0.15


def location_score(location, country, signals):
    text = f"{location} {country}".lower()

    if contains_any(text, PREFERRED_LOCATIONS):
        return 1.0

    if "india" in text:
        return 0.75

    if signals.get("willing_to_relocate", False):
        return 0.60

    return 0.20


def notice_score(signals):
    days = signals.get("notice_period_days", 999)

    try:
        days = float(days)
    except Exception:
        days = 999

    if days <= 30:
        return 1.0
    if days <= 60:
        return 0.72
    if days <= 90:
        return 0.48

    return 0.18


def behavior_score(signals):
    response = float(signals.get("recruiter_response_rate", 0) or 0)
    interview = float(signals.get("interview_completion_rate", 0) or 0)
    offer = float(signals.get("offer_acceptance_rate", 0) or 0)
    github = min(float(signals.get("github_activity_score", 0) or 0) / 10, 1)
    completeness = min(float(signals.get("profile_completeness_score", 0) or 0) / 100, 1)
    saved = min(float(signals.get("saved_by_recruiters_30d", 0) or 0) / 10, 1)
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    verified = 0.5 * int(signals.get("verified_email", False)) + 0.5 * int(signals.get("verified_phone", False))

    return (
        0.24 * response +
        0.18 * interview +
        0.14 * offer +
        0.14 * github +
        0.10 * completeness +
        0.08 * saved +
        0.08 * open_to_work +
        0.04 * verified
    )


def skill_depth_score(skills):
    if not skills:
        return 0.0

    score = 0.0
    max_score = 0.0

    for skill in skills:
        name = str(skill.get("name", "")).lower()
        proficiency = str(skill.get("proficiency", "")).lower()
        months = float(skill.get("duration_months", 0) or 0)
        endorsements = float(skill.get("endorsements", 0) or 0)

        relevance = 0.0

        for keyword, weight in IMPORTANT_SKILLS.items():
            if keyword in name:
                relevance = max(relevance, weight)

        if relevance == 0:
            continue

        proficiency_multiplier = {
            "beginner": 0.35,
            "intermediate": 0.70,
            "advanced": 1.00,
            "expert": 1.10,
        }.get(proficiency, 0.60)

        month_multiplier = min(months / 36.0, 1.0)
        endorsement_multiplier = min(np.log1p(endorsements) / np.log1p(60), 1.0)

        score += relevance * (
            0.45 * proficiency_multiplier +
            0.35 * month_multiplier +
            0.20 * endorsement_multiplier
        )
        max_score += relevance

    return score / max_score if max_score > 0 else 0.0


def ir_nlp_score(text):
    terms = [
        "retrieval", "ranking", "search", "recommendation", "recommender",
        "embedding", "embeddings", "semantic", "vector", "nlp",
        "information retrieval", "hybrid search", "bm25", "faiss",
        "elasticsearch", "opensearch", "pinecone", "milvus", "qdrant", "weaviate",
    ]

    return min(count_any(text, terms) / 8.0, 1.0)


def production_score(text):
    return min(count_any(text, PRODUCTION_TERMS) / 8.0, 1.0)


def eval_score(text):
    terms = [
        "ndcg", "mrr", "map", "a/b", "ab testing",
        "offline evaluation", "online evaluation", "metrics",
        "evaluation framework", "ranking evaluation",
    ]

    return min(count_any(text, terms) / 4.0, 1.0)


def company_score(row):
    career = row["career_history"]

    if not career:
        return 0.35

    consulting_count = 0
    product_signal = 0

    for job in career:
        company = str(job.get("company", "")).lower()
        industry = str(job.get("industry", "")).lower()
        description = str(job.get("description", "")).lower()

        if contains_any(company, CONSULTING_COMPANIES) or "services" in industry or "consulting" in industry:
            consulting_count += 1

        if contains_any(
            description + " " + industry,
            ["product", "platform", "marketplace", "saas", "startup", "users"],
        ):
            product_signal += 1

    if product_signal > 0:
        return 1.0

    if consulting_count == len(career):
        return 0.40

    if consulting_count > 0:
        return 0.68

    return 0.78


def honeypot_penalty(row):
    skills = row["skills"]
    career = row["career_history"]
    text = row["semantic_text"]
    years = float(row["years_experience"] or 0)

    penalty = 1.0

    total_months = sum(float(job.get("duration_months", 0) or 0) for job in career)

    if years > 0 and total_months > 0:
        derived_years = total_months / 12.0

        if abs(derived_years - years) > 4.5:
            penalty *= 0.78

    relevant_count = 0
    advanced_tiny = 0
    advanced_zero = 0

    for skill in skills:
        name = str(skill.get("name", "")).lower()
        proficiency = str(skill.get("proficiency", "")).lower()
        months = float(skill.get("duration_months", 0) or 0)

        is_relevant = any(keyword in name for keyword in IMPORTANT_SKILLS)

        if is_relevant:
            relevant_count += 1

            if proficiency in ["advanced", "expert"] and months < 6:
                advanced_tiny += 1

            if proficiency in ["advanced", "expert"] and months == 0:
                advanced_zero += 1

    if advanced_zero >= 3:
        penalty *= 0.55

    if relevant_count >= 10 and advanced_tiny >= 5:
        penalty *= 0.60

    bad_domain_count = count_any(text, NEGATIVE_DOMAINS)
    ir_count = count_any(text, ["retrieval", "ranking", "search", "recommendation", "nlp", "embedding", "vector"])

    if bad_domain_count >= 4 and ir_count <= 1:
        penalty *= 0.68

    if contains_any(row["title"], BAD_TITLES):
        penalty *= 0.35

    return penalty


def add_feature_scores(df):
    df = df.copy()

    df["title_score"] = df["title"].apply(title_score)
    df["experience_score"] = df["years_experience"].apply(experience_score)
    df["location_score"] = df.apply(lambda row: location_score(row["location"], row["country"], row["signals"]), axis=1)
    df["notice_score"] = df["signals"].apply(notice_score)
    df["behavior_score"] = df["signals"].apply(behavior_score)
    df["skill_depth_score"] = df["skills"].apply(skill_depth_score)
    df["ir_nlp_score"] = df["semantic_text"].apply(ir_nlp_score)
    df["production_score"] = df["semantic_text"].apply(production_score)
    df["eval_score"] = df["semantic_text"].apply(eval_score)
    df["company_score"] = df.apply(company_score, axis=1)
    df["honeypot_penalty"] = df.apply(honeypot_penalty, axis=1)

    return df