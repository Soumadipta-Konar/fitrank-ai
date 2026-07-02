import numpy as np


def select_candidate_pool(df, pool_size=30000):
    df = df.copy()

    df["prefilter_score"] = (
        0.35 * df["lexical_score"] +
        0.18 * df["ir_nlp_score"] +
        0.14 * df["skill_depth_score"] +
        0.10 * df["title_score"] +
        0.08 * df["production_score"] +
        0.06 * df["experience_score"] +
        0.05 * df["behavior_score"] +
        0.04 * df["company_score"]
    )

    pool_size = min(pool_size, len(df))

    top_by_prefilter = df.nlargest(pool_size, "prefilter_score").index

    strong_signal_mask = (
        (df["lexical_score"] >= df["lexical_score"].quantile(0.90)) |
        (df["ir_nlp_score"] >= 0.35) |
        (df["skill_depth_score"] >= 0.35) |
        (df["title_score"] >= 0.68)
    )

    strong_signal_indices = df[strong_signal_mask].index

    pool_indices = np.unique(
        np.concatenate([
            top_by_prefilter.to_numpy(),
            strong_signal_indices.to_numpy()
        ])
    )

    if len(pool_indices) > pool_size:
        temp = df.loc[pool_indices].sort_values(
            "prefilter_score",
            ascending=False
        )
        pool_indices = temp.head(pool_size).index.to_numpy()

    return pool_indices