# Mirrors the official validator checks plus additional self-enforced rules from the architecture spec §13.
import sys
import re
import csv
import gzip
import json
import argparse
import pandas as pd
from pathlib import Path


EXPECTED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CAND_ID_RE = re.compile(r"^CAND_\d{7}$")


# Loads candidate IDs from candidates.jsonl or .jsonl.gz for existence verification.
def _load_candidate_ids(candidates_path: Path) -> set[str]:
    ids = set()
    opener = gzip.open if candidates_path.suffix == ".gz" else open
    with opener(candidates_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cid = rec.get("candidate_id") or rec.get("id")
                if cid:
                    ids.add(str(cid))
            except json.JSONDecodeError:
                pass
    return ids


# Runs all validation checks and returns a list of error strings (empty = pass).
def validate(csv_path: Path, candidates_path: Path | None = None) -> list[str]:
    errors = []
    warnings = []

    try:
        df = pd.read_csv(csv_path, dtype={"candidate_id": str, "rank": int})
    except Exception as e:
        return [f"FATAL: could not parse CSV — {e}"]

    # --- Format checks (mirrors official validator) ---

    # Header
    if list(df.columns) != EXPECTED_HEADER:
        errors.append(f"Header mismatch: got {list(df.columns)}, expected {EXPECTED_HEADER}")

    # Row count
    if len(df) != 100:
        errors.append(f"Row count: got {len(df)}, expected 100")

    # candidate_id format
    bad_ids = df[~df["candidate_id"].str.match(r"^CAND_\d{7}$")]
    if not bad_ids.empty:
        errors.append(f"Invalid candidate_id format in rows: {bad_ids.index.tolist()}")

    # Ranks 1–100 exactly once
    rank_counts = df["rank"].value_counts()
    missing_ranks = set(range(1, 101)) - set(df["rank"])
    dup_ranks = rank_counts[rank_counts > 1].index.tolist()
    if missing_ranks:
        errors.append(f"Missing ranks: {sorted(missing_ranks)}")
    if dup_ranks:
        errors.append(f"Duplicate ranks: {sorted(dup_ranks)}")

    # No duplicate candidate IDs
    dup_ids = df[df["candidate_id"].duplicated(keep=False)]["candidate_id"].unique().tolist()
    if dup_ids:
        errors.append(f"Duplicate candidate_ids: {dup_ids}")

    # Score non-increasing by rank
    df_sorted = df.sort_values("rank").reset_index(drop=True)
    score_violations = []
    for i in range(1, len(df_sorted)):
        if df_sorted.loc[i, "score"] > df_sorted.loc[i - 1, "score"] + 1e-9:
            score_violations.append(
                f"rank {df_sorted.loc[i, 'rank']} score {df_sorted.loc[i, 'score']:.4f} "
                f"> rank {df_sorted.loc[i-1, 'rank']} score {df_sorted.loc[i-1, 'score']:.4f}"
            )
    if score_violations:
        errors.append(f"Score not non-increasing: {score_violations[:5]}")

    # Tie-break: on exact score ties, candidate_id ascending
    tie_violations = []
    for i in range(1, len(df_sorted)):
        prev, curr = df_sorted.loc[i - 1], df_sorted.loc[i]
        if abs(float(prev["score"]) - float(curr["score"])) < 1e-9:
            if curr["candidate_id"] < prev["candidate_id"]:
                tie_violations.append(
                    f"Tie at score {prev['score']:.4f}: "
                    f"{curr['candidate_id']} (rank {curr['rank']}) should come after "
                    f"{prev['candidate_id']} (rank {prev['rank']})"
                )
    if tie_violations:
        errors.append(f"Tie-break violations: {tie_violations}")

    # --- Additional self-enforced checks (§13) ---

    # Reasoning non-empty
    empty_reasoning = df[df["reasoning"].isna() | (df["reasoning"].str.strip() == "")]
    if not empty_reasoning.empty:
        errors.append(f"Empty reasoning in ranks: {empty_reasoning['rank'].tolist()}")

    # No two identical reasonings
    dup_reasoning = df[df["reasoning"].duplicated(keep=False)][["rank", "reasoning"]]
    if not dup_reasoning.empty:
        errors.append(
            f"Duplicate reasoning strings at ranks: "
            f"{dup_reasoning['rank'].tolist()[:10]}"
        )

    # Candidate ID existence check (optional — only if candidates file is provided)
    if candidates_path and candidates_path.exists():
        print(f"  Loading candidate pool from {candidates_path} for ID verification...")
        known_ids = _load_candidate_ids(candidates_path)
        unknown = df[~df["candidate_id"].isin(known_ids)]["candidate_id"].tolist()
        if unknown:
            errors.append(f"candidate_ids not found in pool: {unknown}")
        else:
            print(f"  ✓ All 100 candidate_ids exist in pool ({len(known_ids):,} total)")

    # Tone sanity: flag if a rank > 70 row has zero hedging language
    hedging_terms = [
        "gap", "concern", "limited", "lack", "notice", "consulting", "research",
        "room to", "below", "without", "not yet", "however", "but", "though",
        "moderate", "lower", "shorter", "weaker", "needs", "need", "exceeds", "longer"
    ]
    low_rank_no_hedge = []
    for _, row in df_sorted[df_sorted["rank"] > 70].iterrows():
        reasoning_lower = str(row["reasoning"]).lower()
        if not any(term in reasoning_lower for term in hedging_terms):
            low_rank_no_hedge.append(int(row["rank"]))
    if low_rank_no_hedge:
        warnings.append(
            f"WARN: rank > 70 rows with no hedging language "
            f"(check tone): {low_rank_no_hedge}"
        )

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Validate submission.csv against spec §13 rules."
    )
    parser.add_argument("--csv", default="submission.csv", help="Path to submission CSV")
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl or .jsonl.gz (optional — for ID existence check)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    candidates_path = Path(args.candidates) if args.candidates else None

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    print(f"Validating {csv_path} ...")
    errors, warnings = validate(csv_path, candidates_path)

    for w in warnings:
        print(f"  {w}")

    if errors:
        print(f"\n❌ FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print(f"\n✅ PASSED — all checks clean ({len(warnings)} warning(s))")
        sys.exit(0)


if __name__ == "__main__":
    main()
