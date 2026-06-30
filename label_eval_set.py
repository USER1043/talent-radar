"""
@file Generates a diverse evaluation dataset and automatically assigns baseline relevance scores.
@package precompute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


# Main execution flow for generating the eval set.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=Path("artifacts/features.parquet"), type=Path)
    ap.add_argument("--honeypots", default=Path("artifacts/honeypot_flags.parquet"), type=Path)
    ap.add_argument("--faiss-index", default=Path("artifacts/faiss.index"), type=Path)
    ap.add_argument("--jd-embeds", default=Path("artifacts/jd_embeddings.npy"), type=Path)
    ap.add_argument("--out", default=Path("artifacts/eval_set.csv"), type=Path)
    args = ap.parse_args()

    print("Loading candidate features and honeypot flags...")
    df_feat = pd.read_parquet(args.features)
    df_hp = pd.read_parquet(args.honeypots)
    df = pd.merge(df_feat, df_hp, on="candidate_id")

    print("Loading FAISS index and ideal JD embedding...")
    index = faiss.read_index(str(args.faiss_index))
    jd_embeds = np.load(args.jd_embeds)
    ideal_jd_emb = jd_embeds[1:2].astype("float32")
    faiss.normalize_L2(ideal_jd_emb)

    print("Running semantic search to get similarities...")
    # Get similarities for the top 5000 candidates
    distances, indices = index.search(ideal_jd_emb, 5000)
    sim_dict = {idx: dist for idx, dist in zip(indices[0], distances[0])}

    # Add semantic similarity column
    df["semantic_sim"] = [sim_dict.get(i, 0.0) for i in range(len(df))]

    # Define helper function to detect ML titles
    def is_ml_title(title: str) -> bool:
        t = title.lower()
        has_tech = any(x in t for x in ["ml", "machine learning", "ai", "artificial intelligence", "nlp", "computer vision", "data scientist", "applied scientist"])
        has_bad = any(x in t for x in ["marketing", "hr", "recruiter", "sales", "support", "operations"])
        return has_tech and not has_bad

    print("Sampling candidates for the evaluation set...")
    
    # 1. Honeypots (relevance = 0)
    hp_candidates = df[df["is_honeypot"] == True].sample(n=30, random_state=42).copy()
    hp_candidates["relevance"] = 0
    hp_candidates["sample_type"] = "honeypot"

    # 2. Keyword Stuffers (relevance = 1)
    # Non-ML title in the top 2000 semantic matches, not honeypots
    stuffer_pool = df[
        (df["is_honeypot"] == False) & 
        (~df["current_title"].apply(is_ml_title)) & 
        (df["semantic_sim"] > 0.4)
    ]
    stuffer_candidates = stuffer_pool.sample(n=40, random_state=42).copy()
    stuffer_candidates["relevance"] = 1
    stuffer_candidates["sample_type"] = "keyword_stuffer"

    # 3. Legitimate Strong Fits (relevance = 5)
    # ML title, top semantic matches, not honeypots, not consulting only
    fit_pool = df[
        (df["is_honeypot"] == False) & 
        (df["current_title"].apply(is_ml_title)) & 
        (df["is_consulting_only"] == False) &
        (df["semantic_sim"] > 0.4)
    ]
    fit_candidates = fit_pool.sample(n=40, random_state=42).copy()
    fit_candidates["relevance"] = 5
    fit_candidates["sample_type"] = "strong_fit"

    # 4. Legitimate Mild/Weak Fits (relevance = 2 or 3)
    # ML title, lower semantic similarity, or consulting only
    mid_pool = df[
        (df["is_honeypot"] == False) & 
        (df["current_title"].apply(is_ml_title)) & 
        ((df["is_consulting_only"] == True) | (df["semantic_sim"] <= 0.4))
    ]
    mid_candidates = mid_pool.sample(n=20, random_state=42).copy()
    mid_candidates["relevance"] = 3
    mid_candidates["sample_type"] = "mild_fit"

    # 5. Random background candidates (relevance = 0 or 1)
    bg_pool = df[~df["candidate_id"].isin(
        pd.concat([hp_candidates, stuffer_candidates, fit_candidates, mid_candidates])["candidate_id"]
    )]
    bg_candidates = bg_pool.sample(n=20, random_state=42).copy()
    bg_candidates["relevance"] = [
        1 if is_ml_title(title) else 0 for title in bg_candidates["current_title"]
    ]
    bg_candidates["sample_type"] = "background"

    # Combine all
    eval_df = pd.concat([hp_candidates, stuffer_candidates, fit_candidates, mid_candidates, bg_candidates])
    
    # Keep select columns for human scannability
    scannable_cols = [
        "candidate_id", "relevance", "sample_type", "current_title", 
        "current_company", "years_of_experience", "is_honeypot", 
        "is_consulting_only", "semantic_sim", "skills_json"
    ]
    eval_df = eval_df[scannable_cols]

    # Map skills_json to list of names for readability
    eval_df["skills"] = [
        ", ".join([s["name"] for s in json.loads(sj)]) for sj in eval_df["skills_json"]
    ]
    eval_df = eval_df.drop(columns=["skills_json"])

    # Shuffle eval set
    eval_df = eval_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(args.out, index=False)
    print(f"Wrote {len(eval_df)} labeled pairs to {args.out}")
    print("\nSample Distribution:")
    print(eval_df["sample_type"].value_counts())


if __name__ == "__main__":
    main()
