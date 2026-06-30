"""
Shared schema constants and helpers for the Redrob candidate dataset.
Used by parse_candidates.py, company_stats.py, and honeypot_flags.py.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

DATE_FMT = "%Y-%m-%d"

# JD explicit disqualifier ("Things we explicitly do NOT want" --
# consulting-firms-only career). Extend this list if the real data turns
# up other variants/abbreviations of these names.
CONSULTING_FIRMS = {
    "TCS",
    "Tata Consultancy Services",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
}

HIGH_PROFICIENCY = {"advanced", "expert"}


def parse_date(s: Optional[str]) -> Optional[dt.date]:
    """Parse an ISO (YYYY-MM-DD) date string. Returns None for null/empty/
    unparseable input rather than raising -- the dataset has nulls by
    design (e.g. end_date for current roles)."""
    if not s:
        return None
    try:
        return dt.datetime.strptime(s, DATE_FMT).date()
    except ValueError:
        return None


def months_between(d1: dt.date, d2: dt.date) -> int:
    """Whole calendar months between two dates (d2 - d1). Can be negative
    if d2 is before d1."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def is_consulting_firm(company: str) -> bool:
    return company.strip() in CONSULTING_FIRMS
