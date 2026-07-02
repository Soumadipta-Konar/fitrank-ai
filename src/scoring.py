def add_final_scores(df):
    df = df.copy()

    df["final_score_raw"] = (
        0.28 * df["semantic_score"] +
        0.13 * df["lexical_score"] +
        0.15 * df["ir_nlp_score"] +
        0.12 * df["skill_depth_score"] +
        0.10 * df["production_score"] +
        0.07 * df["title_score"] +
        0.06 * df["experience_score"] +
        0.04 * df["company_score"] +
        0.025 * df["eval_score"] +
        0.015 * df["location_score"] +
        0.01 * df["notice_score"]
    )

    df["behavior_multiplier"] = 0.84 + 0.26 * df["behavior_score"]

    df["final_score_raw"] = (
        df["final_score_raw"] *
        df["behavior_multiplier"] *
        df["honeypot_penalty"]
    )

    # Strong demotions for non-relevant profiles
    df.loc[df["title_score"] <= 0.05, "final_score_raw"] *= 0.35

    # If no retrieval/search/NLP signal, lower confidence
    df.loc[df["ir_nlp_score"] < 0.15, "final_score_raw"] *= 0.78

    # Too junior/senior and weak semantic match
    df.loc[
        (df["experience_score"] < 0.30) & (df["semantic_score"] < 0.40),
        "final_score_raw"
    ] *= 0.70

    df = df.sort_values(
        [
            "final_score_raw",
            "semantic_score",
            "lexical_score",
            "behavior_score",
            "candidate_id"
        ],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return df