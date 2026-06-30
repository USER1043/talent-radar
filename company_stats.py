#!/usr/bin/env python3
"""
Build per-company first-seen statistics from career_history across the
full candidate pool. Used by honeypot_flags.py to catch "tenure predates
the company's existence" honeypots (see ARCHITECTURE.md section 6.2).

Tier A: companies mentioned by >=5 candidates -> we use a LOW PERCENTILE
  (roughly the 10th percentile) of start_date across all mentions as the
  inferred "earliest plausible" date -- not the median, which for small
  samples is just a midpoint (with only 2 mentions, one of them is
  mathematically guaranteed to fall "before" the median, which is a
  coin-flip, not a robustness check). A low percentile with a large
  enough n approximates a lower bound instead.

Tier B: companies mentioned by 1-4 candidates -> not enough independent
  data to trust any statistic, including a percentile. honeypot_flags.py
  does NOT use this script's output for Tier B companies -- it falls back
  to standalone rules instead (skill-count-vs-experience, employment
  overlap, duration-consistency checks) that don't need another
  candidate's data to work.

Usage:
    python company_stats.py \
        --features ./artifacts/features.parquet \
        --out ./artifacts/company_stats.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from schema_utils import parse_date


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_parquet(args.features, columns=["career_history_json"])

    starts_by_company: dict[str, list[int]] = {}
    for career_json in df["career_history_json"]:
        for entry in json.loads(career_json):
            company = (entry.get("company") or "").strip()
            start = parse_date(entry.get("start_date"))
            if not company or start is None:
                continue
            starts_by_company.setdefault(company, []).append(start.toordinal())

    rows = []
    for company, ordinals in starts_by_company.items():
        n = len(ordinals)
        ordinals_sorted = sorted(ordinals)
        # A robust LOWER-BOUND estimate of "earliest plausible" needs a low
        # percentile, not a central one -- the median of a small n is just
        # the midpoint, which guarantees ~half of any 2-candidate group
        # falls "before" it. Use roughly the 10th percentile instead, and
        # require a meaningfully larger n before trusting it at all.
        idx = max(0, int(0.10 * (n - 1)))
        lower_bound_ordinal = ordinals_sorted[idx]
        # Tier A requires >=5 independent mentions -- below that, even a
        # low percentile is too easily skewed by one or two data points,
        # so company_stats.py treats it the same as a single-occurrence
        # (Tier B) company: no reliable cross-check, fall back to
        # standalone rules in honeypot_flags.py instead.
        rows.append(
            {
                "company": company,
                "occurrence_count": n,
                "is_corroborated": n >= 5,
                "earliest_plausible_date": dt.date.fromordinal(lower_bound_ordinal).isoformat(),
            }
        )

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    n_corroborated = int(out_df["is_corroborated"].sum()) if len(out_df) else 0
    print(
        f"Wrote stats for {len(out_df)} distinct companies "
        f"({n_corroborated} corroborated by 5+ candidates / Tier A, "
        f"{len(out_df) - n_corroborated} fewer mentions / Tier B) "
        f"to {args.out}"
    )


if __name__ == "__main__":
    main()
