"""
@file Utility functions for extracting JD rule and behavioral features.
@package precompute
"""

from __future__ import annotations

import datetime as dt
import json
import numpy as np
import pandas as pd


# Returns 1.0 if the candidate has a pure academic/research background, else 0.0.
def compute_is_pure_research(narrative: str) -> float:
    n = narrative.lower()
    research_terms = ["research lab", "research assistant", "phd", "academic", "thesis", "publications", "postdoc"]
    prod_terms = ["production", "deployed", "scale", "pipeline", "infrastructure", "kubernetes", "docker"]
    has_research = any(t in n for t in research_terms)
    has_prod = any(t in n for t in prod_terms)
    return 1.0 if (has_research and not has_prod) else 0.0


# Returns 1.0 if candidate experience is limited only to recent OpenAI wrappers, else 0.0.
def compute_is_recent_wrapper(yoe: float, skills_list: list[str]) -> float:
    skills_lower = [s.lower() for s in skills_list]
    wrapper_skills = ["langchain", "llamaindex", "openai", "prompt engineering"]
    pre_llm_skills = ["scikit-learn", "tensorflow", "pytorch", "pandas", "numpy", "machine learning", "nlp", "computer vision", "opencv", "cnn"]
    has_wrapper = any(s in skills_lower for s in wrapper_skills)
    has_pre_llm = any(s in skills_lower for s in pre_llm_skills)
    return 1.0 if (has_wrapper and not has_pre_llm and yoe < 1.5) else 0.0


# Returns 1.0 if the candidate's current title is non-technical/non-coding, else 0.0.
def compute_no_recent_coding(title: str) -> float:
    t = title.lower()
    non_coding_titles = ["product manager", "project manager", "hr manager", "recruiter", "operations manager", "business analyst", "marketing manager", "graphic designer", "customer support", "sales"]
    return 1.0 if any(x in t for x in non_coding_titles) else 0.0


# Returns 1.0 if candidate is focused on CV/speech/robotics with zero NLP exposure, else 0.0.
def compute_cv_speech_no_nlp(skills_list: list[str]) -> float:
    skills_lower = [s.lower() for s in skills_list]
    cv_speech_robotics = ["computer vision", "opencv", "cnn", "image classification", "object detection", "asr", "tts", "speech recognition", "robotics", "yolo", "gans", "diffusion models"]
    nlp_ir = ["nlp", "nlu", "natural language processing", "information retrieval", "search", "bert", "embeddings", "bm25", "elasticsearch", "milvus", "qdrant", "pinecone", "weaviate", "faiss", "llamaindex", "langchain", "rag", "semantic search"]
    has_cv = any(s in skills_lower for s in cv_speech_robotics)
    has_nlp = any(s in skills_lower for s in nlp_ir)
    return 1.0 if (has_cv and not has_nlp) else 0.0


# Returns 1.0 if the candidate has >5 years of experience but lacks external validation terms, else 0.0.
def compute_no_external_validation(yoe: float, narrative: str) -> float:
    n = narrative.lower()
    validation_terms = ["paper", "publication", "open source", "oss", "github", "talk", "conference", "patent"]
    has_val = any(t in n for t in validation_terms)
    return 1.0 if (yoe > 5.0 and not has_val) else 0.0


# Returns 1.0 if candidate shows a pattern of job-hopping frequently, else 0.0.
def compute_title_chasing(avg_tenure: float, n_employers: int) -> float:
    return 1.0 if (avg_tenure < 18.0 and n_employers >= 2) else 0.0


# Returns 1.0 if candidate has framework wrapper skills but lacks core language/systems skills, else 0.0.
def compute_framework_enthusiast(skills_list: list[str]) -> float:
    skills_lower = [s.lower() for s in skills_list]
    frameworks = ["langchain", "llamaindex", "haystack", "flowise", "autogen"]
    core_skills = ["python", "pytorch", "tensorflow", "scikit-learn", "numpy", "pandas", "data structures", "algorithms", "c++", "go", "java"]
    n_frameworks = sum(1 for s in frameworks if s in skills_lower)
    n_core = sum(1 for s in core_skills if s in skills_lower)
    return 1.0 if (n_frameworks > 0 and n_core == 0) else 0.0


# Returns a composite score (0.0 to 1.0) based on recruiter response, activity, and trust.
def compute_behavioral_score(row: pd.Series) -> float:
    current_date = dt.date(2026, 6, 30)
    active_str = row.get("redrob_last_active_date")
    try:
        active_date = dt.datetime.strptime(active_str, "%Y-%m-%d").date()
        diff_days = max(0, (current_date - active_date).days)
    except (ValueError, TypeError):
        diff_days = 180
    
    recency = np.exp(-np.log(2) * diff_days / 90.0)
    responsiveness = row.get("redrob_recruiter_response_rate") if pd.notna(row.get("redrob_recruiter_response_rate")) else 0.5
    
    resp_time = row.get("redrob_avg_response_time_hours")
    speed = 1.0 / (1.0 + resp_time / 24.0) if (pd.notna(resp_time) and resp_time >= 0) else 0.5
    reliability = row.get("redrob_interview_completion_rate") if pd.notna(row.get("redrob_interview_completion_rate")) else 0.5
    
    offer = row.get("redrob_offer_acceptance_rate")
    offer_signal = offer if (pd.notna(offer) and offer >= 0) else 0.5
    availability = 1.0 if row.get("redrob_open_to_work_flag") == True else 0.6
    
    email = 1.0 if row.get("redrob_verified_email") == True else 0.0
    phone = 1.0 if row.get("redrob_verified_phone") == True else 0.0
    linkedin = 1.0 if row.get("redrob_linkedin_connected") == True else 0.0
    trust = 0.5 * email + 0.3 * phone + 0.2 * linkedin
    
    gh = row.get("redrob_github_activity_score")
    github = gh / 100.0 if (pd.notna(gh) and gh >= 0) else 0.5
    
    return (
        0.15 * recency +
        0.15 * responsiveness +
        0.10 * speed +
        0.15 * reliability +
        0.15 * offer_signal +
        0.10 * availability +
        0.10 * trust +
        0.10 * github
    )


# Returns a penalty score (0.0 to 0.8) based on notice period length.
def compute_notice_period_penalty(notice_days: float) -> float:
    if pd.isna(notice_days) or notice_days <= 30:
        return 0.0
    if notice_days <= 60:
        return 0.2
    if notice_days <= 90:
        return 0.5
    return 0.8


# Extracts all features for a candidate row to feed into the ranking model.
def extract_candidate_features(row: pd.Series) -> dict[str, float]:
    try:
        skills = [s["name"] for s in json.loads(row["skills_json"])]
    except (json.JSONDecodeError, TypeError, KeyError):
        skills = []
        
    yoe = row["years_of_experience"]
    
    feats = {
        "years_of_experience": yoe,
        "n_skills": row["n_skills"],
        "is_consulting_only": 1.0 if row["is_consulting_only"] == True else 0.0,
        "is_pure_research": compute_is_pure_research(row["narrative_text"]),
        "is_recent_wrapper_only": compute_is_recent_wrapper(yoe, skills),
        "no_recent_coding": compute_no_recent_coding(row["current_title"]),
        "cv_speech_robotics_no_nlp": compute_cv_speech_no_nlp(skills),
        "no_external_validation": compute_no_external_validation(yoe, row["narrative_text"]),
        "title_chasing_pattern": compute_title_chasing(row["avg_tenure_months"], row["n_employers"]),
        "framework_enthusiast_signal": compute_framework_enthusiast(skills),
        "behavioral_score": compute_behavioral_score(row),
        "notice_period_penalty": compute_notice_period_penalty(row["redrob_notice_period_days"]),
        "skill_to_experience_ratio": row["skill_to_experience_ratio"]
    }
    return feats
