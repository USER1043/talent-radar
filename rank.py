"""
@file Graded entry point for retrieving, scoring, and ranking candidate profiles.
@package online_ranking
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from build_index import tokenize, extract_jd_texts
from features_utils import extract_candidate_features
from reasoning import generate_reasonings


# Opens a candidate jsonl or jsonl.gz file transparently.
def open_candidates(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# Main execution flow for online candidate ranking.
def main():
    import time
    pipeline_start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--features", default=Path("artifacts/features.parquet"), type=Path)
    ap.add_argument("--honeypots", default=Path("artifacts/honeypot_flags.parquet"), type=Path)
    ap.add_argument("--faiss-index", default=Path("artifacts/faiss.index"), type=Path)
    ap.add_argument("--bm25", default=Path("artifacts/bm25.pkl"), type=Path)
    ap.add_argument("--jd-embeds", default=Path("artifacts/jd_embeddings.npy"), type=Path)
    ap.add_argument("--job-description", default=Path("job_description.docx"), type=Path)
    ap.add_argument("--ranker-model", default=Path("artifacts/ranker_model.pkl"), type=Path)
    args = ap.parse_args()

    print("Stage 1: Loading candidate IDs from candidates file...")
    active_ids = set()
    import re
    cid_re = re.compile(r'"candidate_id"\s*:\s*"(CAND_\d{7})"')
    with open_candidates(args.candidates) as f:
        for line in f:
            m = cid_re.search(line)
            if m:
                active_ids.add(m.group(1))
    print(f"Loaded {len(active_ids)} active candidate IDs.")

    print("Stage 2: Loading cached artifacts...")
    df_feat = pd.read_parquet(args.features)
    df_hp = pd.read_parquet(args.honeypots)
    
    # Filter features to only active candidate IDs
    df_feat = df_feat[df_feat["candidate_id"].isin(active_ids)].reset_index(drop=True)
    df_hp = df_hp[df_hp["candidate_id"].isin(active_ids)].reset_index(drop=True)

    print("Stage 3: Performing hybrid retrieval...")
    # Load JD embedding and FAISS index
    jd_embeds = np.load(args.jd_embeds)
    ideal_jd_emb = jd_embeds[1:2].astype("float32")
    faiss.normalize_L2(ideal_jd_emb)

    index = faiss.read_index(str(args.faiss_index) if hasattr(args, "faiss_index") else "artifacts/faiss.index")
    
    # Retrieve top 1500 semantic matches
    _, faiss_indices = index.search(ideal_jd_emb, 1500)
    faiss_set = set(faiss_indices[0])

    # Retrieve top 1500 lexical matches
    with open(args.bm25, "rb") as f:
        bm25 = pickle.load(f)
    
    literal_jd, _ = extract_jd_texts(args.job_description)
    jd_tokens = tokenize(literal_jd)
    bm25_all_scores = bm25.get_scores(jd_tokens)
    
    bm25_sorted_indices = np.argsort(bm25_all_scores)[::-1]
    bm25_set = set(bm25_sorted_indices[:1500])

    # Union of indices
    shortlist_indices = list(faiss_set.union(bm25_set))
    print(f"Retrieved shortlist size: {len(shortlist_indices)} unique candidates.")

    # Get active shortlist candidates
    shortlist_df = df_feat[df_feat.index.isin(shortlist_indices)].copy()
    shortlist_df = shortlist_df[shortlist_df["candidate_id"].isin(active_ids)].reset_index(drop=True)
    print(f"Active shortlist size after filtering: {len(shortlist_df)} candidates.")

    print("Stage 4: Computing structured candidate features...")
    features_list = []
    for _, row in shortlist_df.iterrows():
        features_list.append(extract_candidate_features(row))
    feats_df = pd.DataFrame(features_list)

    print("Stage 5: Embedding shortlist candidates and computing similarities...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    shortlist_narratives = shortlist_df["narrative_text"].tolist()
    shortlist_embs = model.encode(shortlist_narratives, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(shortlist_embs)

    literal_jd_emb = jd_embeds[0:1].astype("float32")
    faiss.normalize_L2(literal_jd_emb)

    feats_df["semantic_sim_to_jd"] = np.dot(shortlist_embs, literal_jd_emb.T).flatten()
    feats_df["semantic_sim_to_ideal"] = np.dot(shortlist_embs, ideal_jd_emb.T).flatten()

    # Map BM25 scores
    # Maps candidate ID to its index in original df_feat
    feat_id_to_idx = {cid: idx for idx, cid in enumerate(df_feat["candidate_id"])}
    feats_df["bm25_score"] = [bm25_all_scores[feat_id_to_idx[cid]] for cid in shortlist_df["candidate_id"]]

    # Assign features back to shortlist_df for use downstream in reasoning.py
    for col in feats_df.columns:
        shortlist_df[col] = feats_df[col].values

    print("Stage 6: Scoring candidates with trained ranker...")
    with open(args.ranker_model, "rb") as f:
        model_payload = pickle.load(f)
    ranker = model_payload["model"]
    feature_names = model_payload["feature_names"]

    X = feats_df[feature_names].to_numpy()
    shortlist_df["score"] = ranker.predict(X)

    print("Stage 7: Filtering out honeypots...")
    shortlist_df = pd.merge(shortlist_df, df_hp, on="candidate_id")
    non_hp_df = shortlist_df[shortlist_df["is_honeypot"] == False].copy()
    print(f"Candidates remaining after honeypot filtering: {len(non_hp_df)}.")

    print("Stage 8: Sorting, selecting top 100, and formatting output...")
    # Clean score ties with alphabetical candidate_id sort
    # Extracts the numeric part of the ID for robust sorting
    non_hp_df["candidate_id_num"] = non_hp_df["candidate_id"].str.extract(r"(\d+)").astype(int)
    non_hp_df = non_hp_df.sort_values(by=["score", "candidate_id_num"], ascending=[False, True])

    top_100 = non_hp_df.head(100).copy()
    top_100["rank"] = range(1, 101)

    print("Stage 9: Generating reasoning for finalists...")
    top_100["reasoning"] = generate_reasonings(top_100)

    # Output CSV format: candidate_id, rank, score, reasoning
    final_output = top_100[["candidate_id", "rank", "score", "reasoning"]]
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    final_output.to_csv(args.out, index=False)
    print(f"Stage 10: Wrote final ranking list of {len(final_output)} rows to {args.out}")
    print(f"Total pipeline execution time: {time.time() - pipeline_start_time:.2f}s")


if __name__ == "__main__":
    main()
