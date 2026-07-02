# Regression test: asserts the submission's top 10 does not structurally resemble
# the sample_submission.csv keyword-stuffing pattern per architecture spec §13.
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent
SUBMISSION = REPO_ROOT / "submission.csv"
SAMPLE = REPO_ROOT / "sample_submission.csv"

# Job-relevant title keywords — a candidate in the top 10 must have at least one.
JD_TITLE_KEYWORDS = [
    "ml", "machine learning", "ai", "artificial intelligence",
    "nlp", "data science", "deep learning", "research scientist",
    "applied scientist", "engineer", "scientist",
]

# Titles that indicate a clear role mismatch (as seen in sample_submission.csv).
MISMATCH_TITLE_PATTERNS = [
    r"hr\s*manager", r"content\s*writer", r"marketing", r"sales",
    r"recruiter", r"business\s*analyst", r"product\s*manager",
    r"financial\s*analyst", r"accountant", r"operations\s*manager",
]


# Loads the top-N rows from a CSV sorted by rank.
def _top_n(path: Path, n: int = 10) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"candidate_id": str})
    df = df.sort_values("rank").head(n).reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def submission_top10():
    assert SUBMISSION.exists(), f"submission.csv not found at {SUBMISSION}"
    return _top_n(SUBMISSION, 10)


@pytest.fixture(scope="module")
def sample_top10():
    assert SAMPLE.exists(), f"sample_submission.csv not found at {SAMPLE}"
    return _top_n(SAMPLE, 10)


# No candidate in the top 10 should have a reasoning that is a short,
# semicolon-delimited template string (the exact stuffing pattern).
def test_no_template_reasoning_pattern(submission_top10):
    template_re = re.compile(
        r"(HR Manager|Content Writer).+\d+\.\d+ yrs.+\d+ AI core skills",
        re.IGNORECASE,
    )
    violators = []
    for _, row in submission_top10.iterrows():
        if template_re.search(str(row.get("reasoning", ""))):
            violators.append((row["rank"], row["candidate_id"]))
    assert not violators, (
        f"Top-10 contains stuffing-pattern reasoning at ranks: {violators}"
    )


# No candidate in the top 10 should appear in the sample submission's top 10.
def test_no_overlap_with_sample_top10(submission_top10, sample_top10):
    our_ids = set(submission_top10["candidate_id"])
    sample_ids = set(sample_top10["candidate_id"])
    overlap = our_ids & sample_ids
    assert not overlap, (
        f"Top-10 shares {len(overlap)} candidate(s) with the known-bad sample submission: {overlap}"
    )


# Reasoning strings must be LLM-authored sentences, not short template fragments.
def test_reasoning_is_full_sentences(submission_top10):
    short_rows = []
    for _, row in submission_top10.iterrows():
        reasoning = str(row.get("reasoning", "")).strip()
        # A genuine sentence should be > 20 chars and end with punctuation
        if len(reasoning) < 20 or not reasoning[-1] in ".!?":
            short_rows.append((row["rank"], reasoning[:60]))
    assert not short_rows, (
        f"Top-10 reasoning looks like a truncated template: {short_rows}"
    )


# Reasoning strings across the full 100 must all be unique.
def test_no_duplicate_reasonings():
    df = pd.read_csv(SUBMISSION, dtype={"candidate_id": str})
    dupes = df[df["reasoning"].duplicated(keep=False)]["rank"].tolist()
    assert not dupes, f"Duplicate reasoning strings at ranks: {dupes}"


# Scores must be non-increasing across ranks 1–100.
def test_scores_non_increasing():
    df = pd.read_csv(SUBMISSION, dtype={"candidate_id": str}).sort_values("rank")
    scores = df["score"].tolist()
    violations = [
        (int(df.iloc[i]["rank"]), scores[i], int(df.iloc[i - 1]["rank"]), scores[i - 1])
        for i in range(1, len(scores))
        if scores[i] > scores[i - 1] + 1e-9
    ]
    assert not violations, f"Score ordering violated: {violations[:5]}"
