"""
@file Builds search indexes (FAISS and BM25) and precomputes job description embeddings.
@package precompute
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Tokenizes a text into lowercase word tokens.
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# Extracts the literal JD and ideal candidate narrative text from the docx file.
def extract_jd_texts(docx_path: Path) -> tuple[str, str]:
    try:
        import docx
        doc = docx.Document(docx_path)
        all_text = []
        ideal_text_parts = []
        in_ideal = False

        for p in doc.paragraphs:
            txt = p.text.strip()
            if not txt:
                continue
            all_text.append(txt)

            # Start/stop tracking the ideal candidate section
            if "ideal candidate" in txt.lower() and "imagining" in txt.lower():
                in_ideal = True
            elif in_ideal and "disqualifiers" in txt.lower():
                in_ideal = False

            if in_ideal:
                ideal_text_parts.append(txt)

        return "\n".join(all_text), "\n".join(ideal_text_parts)
    except ImportError:
        fallback_jd = "Senior AI/ML Engineer. Applied Machine Learning, search systems, information retrieval, ranking, recommendation."
        fallback_ideal = "Senior AI/ML Engineer with production search/retrieval systems expertise, ranking and recommendation systems."
        return fallback_jd, fallback_ideal


# Main execution flow for building semantic and lexical indexes.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=Path("artifacts/features.parquet"), type=Path)
    ap.add_argument("--job-description", default=Path("job_description.docx"), type=Path)
    ap.add_argument("--out-faiss", default=Path("artifacts/faiss.index"), type=Path)
    ap.add_argument("--out-bm25", default=Path("artifacts/bm25.pkl"), type=Path)
    ap.add_argument("--out-jd-embeds", default=Path("artifacts/jd_embeddings.npy"), type=Path)
    args = ap.parse_args()

    args.out_faiss.parent.mkdir(parents=True, exist_ok=True)

    print("Loading candidate features...")
    df = pd.read_parquet(args.features, columns=["candidate_id", "narrative_text"])

    print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print("Embedding candidate narrative texts...")
    embeddings = model.encode(
        df["narrative_text"].tolist(),
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print("Building and saving FAISS index...")
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)
    faiss.write_index(index, str(args.out_faiss))
    print(f"Saved FAISS index to {args.out_faiss}")

    print("Tokenizing corpus and building BM25 index...")
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [tokenize(doc) for doc in df["narrative_text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(args.out_bm25, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index to {args.out_bm25}")

    print("Extracting and embedding Job Description...")
    literal_jd, ideal_jd = extract_jd_texts(args.job_description)
    print(f"Ideal JD narrative length: {len(ideal_jd)} chars")
    
    jd_embeddings = model.encode([literal_jd, ideal_jd], convert_to_numpy=True)
    np.save(args.out_jd_embeds, jd_embeddings)
    print(f"Saved JD embeddings to {args.out_jd_embeds}")


if __name__ == "__main__":
    main()
