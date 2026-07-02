"""Unit tests for verifying the Streamlit application's data loading."""

import sys
from pathlib import Path

# Add project root and app directory to sys.path for importing modules.
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "app"))


# Asserts that the loaded dataframe contains all fields needed by streamlit_app.py.
def test_streamlit_load_data():
    from app import load_data
    df = load_data.__wrapped__()
    
    required_cols = [
        "candidate_id",
        "rank",
        "score",
        "reasoning",
        "behavioral_score",
        "is_consulting_only",
        "is_pure_research",
        "is_recent_wrapper_only",
        "no_recent_coding"
    ]
    
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' is missing from the loaded dataframe"
