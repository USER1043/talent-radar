"""
@file Trains and compares LightGBM LambdaMART ranker vs Ridge regression on evaluation set.
@package precompute
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import lightgbm as lgb

from build_index import tokenize, extract_jd_texts
from features_utils import extract_candidate_features


# Calculates Normalized Discounted Cumulative Gain at K.
def ndcg_k(relevances: list[float], k: int) -> float:
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k]))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(relevances, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


# Calculates Average Precision for binary relevance (relevance >= 3 is relevant).
def average_precision(relevances: list[float]) -> float:
    binary_rel = [1.0 if r >= 3 else 0.0 for r in relevances]
    num_relevant = sum(binary_rel)
    if num_relevant == 0:
        return 0.0
    ap = 0.0
    hits = 0.0
    for i, r in enumerate(binary_rel):
        if r == 1:
            hits += 1
            ap += hits / (i + 1)
    return ap / num_relevant


# Calculates Precision at 10.
def precision_at_10(relevances: list[float]) -> float:
    binary_rel = [1.0 if r >= 3 else 0.0 for r in relevances[:10]]
    return sum(binary_rel) / 10.0


# Computes composite hackathon scoring formula.
def compute_composite_score(ndcg10: float, ndcg50: float, map_score: float, p10: float) -> float:
    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_score + 0.05 * p10


# Main execution flow for training the ranking models.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=Path("artifacts/eval_set.csv"), type=Path)
    ap.add_argument("--features", default=Path("artifacts/features.parquet"), type=Path)
    ap.add_argument("--bm25", default=Path("artifacts/bm25.pkl"), type=Path)
    ap.add_argument("--jd-embeds", default=Path("artifacts/jd_embeddings.npy"), type=Path)
    ap.add_argument("--job-description", default=Path("job_description.docx"), type=Path)
    ap.add_argument("--out-model", default=Path("artifacts/ranker_model.pkl"), type=Path)
    args = ap.parse_args()

    print("Loading eval set and features parquet...")
    eval_df = pd.read_csv(args.eval_set)
    df_feat = pd.read_parquet(args.features)

    # Merge features to eval candidates
    merged_df = pd.merge(eval_df[["candidate_id", "relevance"]], df_feat, on="candidate_id")

    print("Extracting structured candidate features...")
    features_list = []
    for _, row in merged_df.iterrows():
        features_list.append(extract_candidate_features(row))
    X_df = pd.DataFrame(features_list)

    print("Computing semantic similarities to JD versions...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    eval_narratives = merged_df["narrative_text"].tolist()
    eval_embs = model.encode(eval_narratives, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(eval_embs)

    jd_embeds = np.load(args.jd_embeds)
    literal_jd_emb = jd_embeds[0:1].astype("float32")
    ideal_jd_emb = jd_embeds[1:2].astype("float32")
    faiss.normalize_L2(literal_jd_emb)
    faiss.normalize_L2(ideal_jd_emb)

    X_df["semantic_sim_to_jd"] = np.dot(eval_embs, literal_jd_emb.T).flatten()
    X_df["semantic_sim_to_ideal"] = np.dot(eval_embs, ideal_jd_emb.T).flatten()

    print("Loading BM25 and computing lexical scores...")
    with open(args.bm25, "rb") as f:
        bm25 = pickle.load(f)
    literal_jd, _ = extract_jd_texts(args.job_description)
    jd_tokens = tokenize(literal_jd)
    bm25_all_scores = bm25.get_scores(jd_tokens)

    id_to_idx = {cid: idx for idx, cid in enumerate(df_feat["candidate_id"])}
    X_df["bm25_score"] = [bm25_all_scores[id_to_idx[cid]] for cid in merged_df["candidate_id"]]

    # Features and labels
    feature_names = list(X_df.columns)
    X = X_df.to_numpy()
    y = merged_df["relevance"].to_numpy()

    print(f"Features: {feature_names}")
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lgb_scores = []
    ridge_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 1. Ridge regression
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_preds = ridge.predict(X_test)
        
        # Sort test indices by predictions
        sorted_test_idx = np.argsort(ridge_preds)[::-1]
        ridge_rel = y_test[sorted_test_idx].tolist()
        
        r_n10 = ndcg_k(ridge_rel, 10)
        r_n50 = ndcg_k(ridge_rel, 50)
        r_map = average_precision(ridge_rel)
        r_p10 = precision_at_10(ridge_rel)
        r_comp = compute_composite_score(r_n10, r_n50, r_map, r_p10)
        ridge_scores.append(r_comp)

        # 2. LightGBM LambdaRank
        # LambdaRank requires sorting inputs by group, but CV creates arbitrary indices.
        # We sort them by index so it's a single contiguous group.
        lgb_ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            random_state=42,
            verbosity=-1
        )
        lgb_ranker.fit(
            X_train, y_train,
            group=[len(X_train)]
        )
        lgb_preds = lgb_ranker.predict(X_test)
        
        sorted_lgb_idx = np.argsort(lgb_preds)[::-1]
        lgb_rel = y_test[sorted_lgb_idx].tolist()
        
        l_n10 = ndcg_k(lgb_rel, 10)
        l_n50 = ndcg_k(lgb_rel, 50)
        l_map = average_precision(lgb_rel)
        l_p10 = precision_at_10(lgb_rel)
        l_comp = compute_composite_score(l_n10, l_n50, l_map, l_p10)
        lgb_scores.append(l_comp)

    mean_ridge = np.mean(ridge_scores)
    mean_lgb = np.mean(lgb_scores)

    print(f"\nModel Performance (5-Fold CV Composite Score):")
    print(f"  Ridge Regression: {mean_ridge:.4f}")
    print(f"  LightGBM LambdaRank: {mean_lgb:.4f}")

    # Save winner
    if mean_lgb >= mean_ridge:
        print("\nLightGBM LambdaRank wins! Training on full eval set...")
        winner = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=50,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=7,
            random_state=42,
            verbosity=-1
        )
        winner.fit(X, y, group=[len(X)])
        
        # Display Feature Importance
        importances = winner.feature_importances_
        sorted_imp_idx = np.argsort(importances)[::-1]
        print("\nFeature Importances:")
        for idx in sorted_imp_idx:
            print(f"  {feature_names[idx]}: {importances[idx]}")
            
        model_payload = {
            "type": "lightgbm",
            "model": winner,
            "feature_names": feature_names
        }
    else:
        print("\nRidge Regression wins! Training on full eval set...")
        winner = Ridge(alpha=1.0)
        winner.fit(X, y)
        
        # Display Coefficients
        coefs = winner.coef_
        sorted_coef_idx = np.argsort(np.abs(coefs))[::-1]
        print("\nModel Coefficients:")
        for idx in sorted_coef_idx:
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
            
        model_payload = {
            "type": "ridge",
            "model": winner,
            "feature_names": feature_names
        }

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "wb") as f:
        pickle.dump(model_payload, f)
    print(f"\nSaved winning model payload to {args.out_model}")


if __name__ == "__main__":
    main()
