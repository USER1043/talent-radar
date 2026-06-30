#!/usr/bin/env python3
"""
Stream-parse candidates.jsonl(.gz) into a flat feature table (features.parquet).

Run this once, offline -- it has no time limit. Never loads all 100K raw
records into memory at once; reads line-by-line and writes to parquet in
batches.

Usage:
    python parse_candidates.py \
        --candidates ../candidates.jsonl.gz \
        --out ./artifacts/features.parquet

Nested structures that downstream scripts need in full (career_history,
skills) are kept as JSON strings in their own columns rather than expanded
into hundreds of mostly-empty columns -- company_stats.py and
honeypot_flags.py re-parse them as needed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from schema_utils import is_consulting_firm, months_between, parse_date

BATCH_SIZE = 5000

# All 23 redrob_signals fields except the two structured ones
# (expected_salary_range_inr_lpa, skill_assessment_scores) which get their
# own handling below.
SIMPLE_SIGNAL_FIELDS = (
    "profile_completeness_score", "signup_date", "last_active_date",
    "open_to_work_flag", "profile_views_received_30d",
    "applications_submitted_30d", "recruiter_response_rate",
    "avg_response_time_hours", "connection_count", "endorsements_received",
    "notice_period_days", "preferred_work_mode", "willing_to_relocate",
    "github_activity_score", "search_appearance_30d",
    "saved_by_recruiters_30d", "interview_completion_rate",
    "offer_acceptance_rate", "verified_email", "verified_phone",
    "linkedin_connected",
)


def open_candidates(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_records(path: Path) -> Iterator[dict]:
    with open_candidates(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten(record: dict) -> dict:
    profile = record["profile"]
    career = record.get("career_history", []) or []
    skills = record.get("skills", []) or []
    signals = record.get("redrob_signals", {}) or {}
    education = record.get("education", []) or []

    yoe = profile.get("years_of_experience") or 0.0

    n_employers = len({c.get("company", "") for c in career})
    total_tenure_months = sum(c.get("duration_months", 0) or 0 for c in career)
    avg_tenure_months = total_tenure_months / n_employers if n_employers else 0.0

    # Internal data-integrity check, also doubles as a honeypot signal:
    # does declared duration_months roughly match the start/end date gap?
    # Only checked for completed roles (current roles have no end_date).
    consistency_mismatches = 0
    for c in career:
        if c.get("is_current"):
            continue
        start = parse_date(c.get("start_date"))
        end = parse_date(c.get("end_date"))
        if start is None or end is None:
            continue
        computed_months = months_between(start, end)
        declared_months = c.get("duration_months", 0) or 0
        if abs(computed_months - declared_months) > 2:  # 2-month tolerance
            consistency_mismatches += 1

    is_consulting_only = bool(career) and all(
        is_consulting_firm(c.get("company", "")) for c in career
    )
    has_product_company_experience = any(
        not is_consulting_firm(c.get("company", "")) for c in career
    )

    n_skills = len(skills)
    n_expert_or_advanced_skills = sum(
        1 for s in skills if s.get("proficiency") in ("advanced", "expert")
    )
    n_zero_duration_high_proficiency_skills = sum(
        1
        for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and (s.get("duration_months", 0) or 0) == 0
    )
    skill_to_experience_ratio = n_skills / max(yoe, 0.5)

    narrative_parts = [profile.get("headline", ""), profile.get("summary", "")]
    narrative_parts += [c.get("description", "") for c in career]
    narrative_parts += [s.get("name", "") for s in skills]
    narrative_text = " ".join(p for p in narrative_parts if p)

    education_tier = education[0].get("tier", "unknown") if education else "unknown"

    flat = {
        "candidate_id": record["candidate_id"],
        "years_of_experience": yoe,
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "current_company_size": profile.get("current_company_size", ""),
        "current_industry": profile.get("current_industry", ""),
        "location": profile.get("location", ""),
        "country": profile.get("country", ""),
        "n_career_entries": len(career),
        "n_employers": n_employers,
        "total_tenure_months": total_tenure_months,
        "avg_tenure_months": avg_tenure_months,
        "tenure_consistency_mismatches": consistency_mismatches,
        "is_consulting_only": is_consulting_only,
        "has_product_company_experience": has_product_company_experience,
        "n_skills": n_skills,
        "n_expert_or_advanced_skills": n_expert_or_advanced_skills,
        "n_zero_duration_high_proficiency_skills": n_zero_duration_high_proficiency_skills,
        "skill_to_experience_ratio": skill_to_experience_ratio,
        "education_tier": education_tier,
        "narrative_text": narrative_text,
        # full structures, re-parsed downstream where needed
        "career_history_json": json.dumps(career),
        "skills_json": json.dumps(skills),
    }

    for key in SIMPLE_SIGNAL_FIELDS:
        flat[f"redrob_{key}"] = signals.get(key)

    salary = signals.get("expected_salary_range_inr_lpa") or {}
    flat["redrob_expected_salary_min"] = salary.get("min")
    flat["redrob_expected_salary_max"] = salary.get("max")
    flat["redrob_skill_assessment_scores_json"] = json.dumps(
        signals.get("skill_assessment_scores") or {}
    )

    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    batch = []
    n_total = 0

    def flush(rows):
        nonlocal writer, n_total
        if not rows:
            return
        # NOTE: if a batch happens to have every value of some nullable
        # column as None, pyarrow may infer a different dtype than a
        # previous batch where that column had real values, and
        # write_table will raise a schema-mismatch error. Hasn't been
        # hit in testing on the 50-record sample, but if it shows up on
        # the full 100K, the fix is to pass an explicit pa.schema(...)
        # to both from_pandas() and ParquetWriter() instead of inferring.
        table = pa.Table.from_pandas(pd.DataFrame(rows))
        if writer is None:
            writer = pq.ParquetWriter(args.out, table.schema)
        writer.write_table(table)
        n_total += len(rows)

    for record in iter_records(args.candidates):
        batch.append(flatten(record))
        if len(batch) >= BATCH_SIZE:
            flush(batch)
            batch = []
    flush(batch)
    if writer is not None:
        writer.close()

    print(f"Wrote {n_total} candidate records to {args.out}")


if __name__ == "__main__":
    main()
