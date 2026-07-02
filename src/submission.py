import subprocess
import sys
from pathlib import Path

import numpy as np

from src.features import IMPORTANT_SKILLS


def top_skills_for_reason(skills):
    priority = list(IMPORTANT_SKILLS.keys())
    selected = []

    for skill in skills:
        name = str(skill.get("name", ""))
        low = name.lower()

        if any(keyword in low for keyword in priority):
            selected.append(name)

    seen = set()
    output = []

    for skill in selected:
        key = skill.lower()

        if key not in seen:
            output.append(skill)
            seen.add(key)

    return output[:4]


def make_reason(row, semantic_95, semantic_85):
    parts = []

    title = row["title"] if str(row["title"]).strip() else "Candidate"
    parts.append(f"{title} with {row['years_experience']:.1f} years experience")

    skills = top_skills_for_reason(row["skills"])

    if skills:
        parts.append("relevant skills: " + ", ".join(skills))

    if row["semantic_score"] >= semantic_95:
        parts.append("very strong semantic match to the JD")
    elif row["semantic_score"] >= semantic_85:
        parts.append("strong semantic match to the JD")

    if row["ir_nlp_score"] >= 0.65:
        parts.append("strong search/retrieval/ranking signal")
    elif row["ir_nlp_score"] >= 0.35:
        parts.append("moderate NLP/IR signal")

    if row["production_score"] >= 0.60:
        parts.append("clear production/system ownership")
    elif row["production_score"] >= 0.30:
        parts.append("some production engineering exposure")

    if row["eval_score"] >= 0.50:
        parts.append("mentions ranking/evaluation metrics")

    if row["behavior_score"] >= 0.70:
        parts.append("strong Redrob engagement signals")
    elif row["behavior_score"] < 0.35:
        parts.append("weaker availability/engagement signals")

    notice = row["signals"].get("notice_period_days", None)

    if notice is not None:
        if notice <= 30:
            parts.append("short notice period")
        elif notice > 90:
            parts.append("long notice period concern")

    if row["location_score"] >= 0.75 and str(row["location"]).strip():
        parts.append(f"logistically relevant location: {row['location']}")

    if row["honeypot_penalty"] < 0.8:
        parts.append("ranked conservatively due to consistency risk")

    return "; ".join(parts)[:300]


def create_submission(df, out_path, debug_path=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    top100 = df.head(100).copy()

    raw = top100["final_score_raw"].values
    norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    top100["score"] = (0.500000 + 0.495000 * norm).round(6)

    top100 = top100.sort_values(
        ["score", "final_score_raw", "candidate_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    top100["rank"] = np.arange(1, 101)

    semantic_95 = df["semantic_score"].quantile(0.95)
    semantic_85 = df["semantic_score"].quantile(0.85)

    top100["reasoning"] = top100.apply(
        lambda row: make_reason(row, semantic_95, semantic_85),
        axis=1,
    )

    submission = top100[["candidate_id", "rank", "score", "reasoning"]]
    submission.to_csv(out_path, index=False)

    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        debug_cols = [
            "candidate_id", "rank", "score", "title", "years_experience",
            "location", "country", "final_score_raw", "semantic_score",
            "lexical_score", "ir_nlp_score", "skill_depth_score",
            "production_score", "behavior_score", "honeypot_penalty",
            "reasoning",
        ]

        top100[debug_cols].to_csv(debug_path, index=False)

    return submission


def validate_submission(validator_path, submission_path):
    result = subprocess.run(
        [sys.executable, str(validator_path), str(submission_path)],
        capture_output=True,
        text=True,
    )

    return result.stdout, result.stderr