#!/usr/bin/env python3
"""
Flag honeypot candidates using OR-logic across independent rules (see
ARCHITECTURE.md section 6). Any single rule firing flags the candidate --
this deliberately avoids averaging risk scores, which would let one
strong red flag get diluted by other normal-looking signals and slip
under a threshold.

Usage:
    python honeypot_flags.py \
        --features ./artifacts/features.parquet \
        --company-stats ./artifacts/company_stats.parquet \
        --out ./artifacts/honeypot_flags.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from schema_utils import parse_date

# AGENT TODO: these thresholds are starting points. Before relying on
# them, plot the real distribution of years_of_experience and n_skills
# across the actual 100K pool (the ~80 honeypots should show up as a
# clear outlier cluster) and adjust.
LOW_EXPERIENCE_YEARS = 1.0      # "0 experience" in the spec's example, with slack
HIGH_SKILL_COUNT = 8            # "excessive skills" in the spec's example
COMPANY_FOUNDING_BUFFER_MONTHS = 6  # tolerance before flagging "predates founding"
OVERLAP_TOLERANCE_MONTHS = 2  # small overlaps (notice-period handover) are normal


def has_overlapping_entries(career: list[dict]) -> bool:
    """True if two non-current roles' date ranges overlap by more than
    OVERLAP_TOLERANCE_MONTHS."""
    spans = []
    for entry in career:
        if entry.get("is_current"):
            continue
        start = parse_date(entry.get("start_date"))
        end = parse_date(entry.get("end_date"))
        if start is None or end is None:
            continue
        spans.append((start, end))
    spans.sort(key=lambda s: s[0])
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        if s2 < e1:
            overlap_months = (e1.year - s2.year) * 12 + (e1.month - s2.month)
            if overlap_months > OVERLAP_TOLERANCE_MONTHS:
                return True
    return False


def check_candidate(row, company_stats: dict) -> tuple[bool, list[str]]:
    reasons = []

    # Rule 1 -- expert/advanced skill claimed with zero time spent on it
    if row["n_zero_duration_high_proficiency_skills"] >= 1:
        reasons.append("expert/advanced skill with 0 months duration")

    # Rule 2 -- excessive skills claimed alongside near-zero experience.
    # This is the literal pattern the spec describes ("excessive skills
    # with 0 experience"), rather than a ratio -- a ratio divides by
    # years_of_experience, which over-penalizes genuinely skilled junior
    # candidates instead of catching the actual trap.
    if row["years_of_experience"] <= LOW_EXPERIENCE_YEARS and row["n_skills"] >= HIGH_SKILL_COUNT:
        reasons.append("excessive skill count alongside near-zero experience")

    career = json.loads(row["career_history_json"])

    # Rule 3 -- tenure predates the company's inferred founding.
    # Tier A only (corroborated companies, n>=5); Tier B companies have no
    # reliable independent data to check against, so they fall through to
    # rules 1/2/4/5 instead -- see company_stats.py docstring.
    for entry in career:
        company = (entry.get("company") or "").strip()
        stat = company_stats.get(company)
        if stat is None or not stat.get("is_corroborated"):
            continue
        start = parse_date(entry.get("start_date"))
        earliest_plausible = parse_date(stat.get("earliest_plausible_date"))
        if start and earliest_plausible:
            buffer_days = COMPANY_FOUNDING_BUFFER_MONTHS * 30
            if (earliest_plausible - start).days > buffer_days:
                reasons.append(f"tenure at '{company}' predates its inferred founding")
                break

    # Rule 4 -- overlapping full-time roles
    if has_overlapping_entries(career):
        reasons.append("overlapping employment dates")

    # Rule 5 -- duration_months inconsistent with declared start/end dates
    if row["tenure_consistency_mismatches"] > 0:
        reasons.append("duration_months inconsistent with start/end dates")

    # Rule 6 -- education sequence is logically impossible
    DEGREE_RANK = {
        "phd": 3, "me": 2, "mtech": 2, "mba": 2, "msc": 2,
        "be": 1, "btech": 1, "bsc": 1, "ba": 1
    }
    try:
        education = json.loads(row["education_json"])
    except Exception:
        education = []
        
    for edu1 in education:
        deg1 = (edu1.get("degree") or "").lower().replace(".", "").strip()
        rank1 = DEGREE_RANK.get(deg1)
        end1 = edu1.get("end_year")
        if rank1 is None or end1 is None:
            continue
            
        for edu2 in education:
            deg2 = (edu2.get("degree") or "").lower().replace(".", "").strip()
            rank2 = DEGREE_RANK.get(deg2)
            start2 = edu2.get("start_year")
            if rank2 is None or start2 is None:
                continue
                
            if rank1 > rank2 and end1 < start2:
                reasons.append("logically impossible education sequence")
                break
        if "logically impossible education sequence" in reasons:
            break

    return (len(reasons) > 0), reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--company-stats", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    features = pd.read_parquet(args.features)
    stats_df = pd.read_parquet(args.company_stats)
    company_stats = stats_df.set_index("company").to_dict(orient="index")

    flags, reasons_list = [], []
    # NOTE: iterrows() is fine for an offline, untimed precompute step on
    # 100K rows (seconds to low minutes). If this script ever needs to
    # move into the timed rank.py path, vectorize it first.
    for _, row in features.iterrows():
        flagged, reasons = check_candidate(row, company_stats)
        flags.append(flagged)
        reasons_list.append("; ".join(reasons))

    out_df = pd.DataFrame(
        {
            "candidate_id": features["candidate_id"],
            "is_honeypot": flags,
            "honeypot_reasons": reasons_list,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    rate = out_df["is_honeypot"].mean() if len(out_df) else 0.0
    print(
        f"Flagged {int(out_df['is_honeypot'].sum())} / {len(out_df)} candidates "
        f"as honeypots ({rate:.2%})."
    )


if __name__ == "__main__":
    main()
