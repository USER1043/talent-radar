"""
@file Deterministic reasoning generation pipeline for candidate ranking.
@package online_ranking
"""

from __future__ import annotations

import json
import re
import hashlib
import pandas as pd
from dataclasses import dataclass


@dataclass
class CandidateAnalysis:
    """Normalized candidate feature representation for rule-based planning."""
    candidate_id: str
    yoe: float
    title: str
    company: str
    semantic_match: float
    behavior_score: float
    notice_days: float
    has_recent_coding: bool
    consulting_only: bool
    open_source: bool
    publications: bool
    leadership: bool


@dataclass
class ReasonPlan:
    """Recruiter reasoning plan containing planned drivers and differentiator."""
    candidate_id: str
    persona: str
    primary_reason: str
    secondary_reason: str
    concern: str | None
    tone: str
    confidence: str
    evidence: list[str]
    differentiator: str | None


# Maps specific titles to leadership flag.
def _check_leadership(title: str) -> bool:
    t = title.lower()
    return any(x in t for x in ["lead", "staff", "principal", "manager", "director", "head", "architect"])


# Creates a normalized CandidateAnalysis dataclass representation from a pandas row.
def analyze_candidate(row: pd.Series) -> CandidateAnalysis:
    title = str(row.get("current_title", "Engineer")).strip()
    company = str(row.get("current_company", "Private")).strip()
    narrative = str(row.get("narrative_text", "")).lower()
    
    yoe = float(row.get("years_of_experience", 0.0))
    semantic_match = float(row.get("semantic_sim_to_ideal", 0.5))
    behavior_score = float(row.get("behavioral_score", 0.5))
    notice_days = float(row.get("redrob_notice_period_days", 0.0))
    
    has_recent_coding = float(row.get("no_recent_coding", 0.0)) == 0.0
    consulting_only = float(row.get("is_consulting_only", 0.0)) == 1.0
    
    # Check open source and publication cues in the narrative
    open_source = any(x in narrative for x in ["open source", "oss", "github", "contributions"])
    publications = any(x in narrative for x in ["publication", "paper", "patent", "thesis"])
    leadership = _check_leadership(title)
    
    return CandidateAnalysis(
        candidate_id=row["candidate_id"],
        yoe=yoe,
        title=title,
        company=company,
        semantic_match=semantic_match,
        behavior_score=behavior_score,
        notice_days=notice_days,
        has_recent_coding=has_recent_coding,
        consulting_only=consulting_only,
        open_source=open_source,
        publications=publications,
        leadership=leadership,
    )


# Evaluates candidate features against thresholds to determine planned reasons, tone, confidence, and evidence.
def plan_reason(analysis: CandidateAnalysis, next_analysis: CandidateAnalysis | None) -> ReasonPlan:
    # 1. Persona Classification
    if analysis.yoe >= 5.0 and analysis.semantic_match >= 0.58 and analysis.behavior_score >= 0.70 and analysis.notice_days <= 30:
        persona = "Elite Match"
    elif analysis.has_recent_coding is False or analysis.behavior_score < 0.55 or analysis.notice_days > 90:
        persona = "High Risk"
    elif analysis.yoe >= 4.0 and not analysis.consulting_only and analysis.semantic_match >= 0.55 and analysis.has_recent_coding:
        persona = "Strong Product Engineer"
    elif analysis.yoe >= 7.0 and analysis.semantic_match >= 0.55:
        persona = "Senior Specialist"
    elif analysis.leadership and analysis.yoe >= 5.0:
        persona = "Leadership Profile"
    elif analysis.publications:
        persona = "Research-Oriented"
    elif analysis.notice_days <= 15 or (analysis.notice_days <= 30 and analysis.behavior_score >= 0.70):
        persona = "Fast Hire"
    elif analysis.consulting_only:
        persona = "Consulting Background"
    elif analysis.yoe < 4.0 and analysis.semantic_match >= 0.53:
        persona = "Emerging Candidate"
    else:
        persona = "Backend Generalist"

    # 2. Primary Reason Selection
    if analysis.semantic_match >= 0.58:
        primary_reason = "Exceptional JD alignment"
    elif not analysis.consulting_only and analysis.yoe >= 5.0:
        primary_reason = "Strong production engineering background"
    elif analysis.leadership and analysis.yoe >= 5.0:
        primary_reason = "Technical leadership"
    elif analysis.notice_days <= 15:
        primary_reason = "Hiring readiness"
    elif analysis.consulting_only:
        primary_reason = "Relevant experience with delivery-focused background"
    else:
        primary_reason = "Core ML/AI technical competence"

    # 3. Secondary Reason Selection
    if analysis.yoe >= 7.0:
        secondary_reason = "years of experience"
    elif analysis.company.lower() in ["google", "apple", "microsoft", "amazon", "netflix", "adobe", "meta", "salesforce"]:
        secondary_reason = "top-tier company pedigree"
    elif analysis.has_recent_coding:
        secondary_reason = "active hands-on coding"
    elif analysis.open_source:
        secondary_reason = "open source contributions"
    elif analysis.publications:
        secondary_reason = "research publications"
    elif analysis.behavior_score >= 0.70:
        secondary_reason = "high recruiter responsiveness"
    else:
        secondary_reason = "solid candidate engagement"

    # 4. Concern Selection
    if not analysis.has_recent_coding:
        concern = "Lack of recent coding"
    elif analysis.notice_days > 30:
        concern = "Long notice period"
    elif analysis.consulting_only:
        concern = "Consulting-heavy experience"
    elif analysis.semantic_match < 0.55:
        concern = "Lower semantic alignment"
    elif not analysis.open_source and not analysis.publications and analysis.yoe >= 5.0:
        concern = "Missing external validation"
    else:
        concern = None

    # 5. Tone Classification
    if persona == "Elite Match":
        tone = "Outstanding"
    elif persona in ["Strong Product Engineer", "Senior Specialist", "Leadership Profile"] and concern is None:
        tone = "Strong"
    elif concern is None:
        tone = "Positive"
    elif persona == "High Risk":
        tone = "Weak"
    elif analysis.notice_days > 60 or not analysis.has_recent_coding:
        tone = "Cautious"
    else:
        tone = "Balanced"

    # 6. Confidence Classification
    if analysis.semantic_match >= 0.57 and analysis.yoe >= 5.0 and analysis.notice_days <= 30:
        confidence = "High"
    elif persona == "High Risk" or (analysis.semantic_match < 0.52 and analysis.notice_days > 60):
        confidence = "Low"
    else:
        confidence = "Medium"

    # 7. Evidence Selection (2 to 4 items)
    evidence = [
        f"{analysis.yoe:.1f} years",
        f"{analysis.title} at {analysis.company}",
    ]
    if analysis.semantic_match >= 0.55:
        evidence.append(f"strong search systems background ({analysis.semantic_match:.2%})")
    if analysis.open_source:
        evidence.append("active GitHub presence")
    if analysis.publications:
        evidence.append("academic publication track record")
    if analysis.behavior_score >= 0.70:
        evidence.append("high recruiter responsiveness")

    # 8. Relative Differentiator Awareness
    differentiator = None
    if next_analysis is not None:
        if analysis.notice_days < next_analysis.notice_days - 15:
            differentiator = "shorter hiring timeline"
        elif analysis.semantic_match > next_analysis.semantic_match + 0.02:
            differentiator = "closer semantic match to search requirements"
        elif analysis.behavior_score > next_analysis.behavior_score + 0.05:
            differentiator = "higher platform engagement"
        elif analysis.yoe > next_analysis.yoe + 2.0:
            differentiator = "greater professional experience"
        elif analysis.has_recent_coding and not next_analysis.has_recent_coding:
            differentiator = "active hands-on coding role"
        elif not analysis.consulting_only and next_analysis.consulting_only:
            differentiator = "product company background"

    return ReasonPlan(
        candidate_id=analysis.candidate_id,
        persona=persona,
        primary_reason=primary_reason,
        secondary_reason=secondary_reason,
        concern=concern,
        tone=tone,
        confidence=confidence,
        evidence=evidence,
        differentiator=differentiator,
    )


# Generates a natural language recruiter-style summary based solely on the ReasonPlan details.
def generate_summary_from_plan(plan: ReasonPlan) -> str:
    # Seed deterministic choices based on candidate_id hash
    seed = int(hashlib.md5(plan.candidate_id.encode()).hexdigest(), 16) % 100
    
    yoe_desc = plan.evidence[0]
    role_desc = plan.evidence[1]
    
    # Extra strengths
    extra_strengths = plan.evidence[2:] if len(plan.evidence) > 2 else []
    strength_phrase = f" and {extra_strengths[0]}" if extra_strengths else ""
    
    differentiator_clause = f" They stand out with a {plan.differentiator} compared to adjacent candidates." if plan.differentiator else ""
    concern_clause = f" However, hiring is constrained by {plan.concern.lower()}." if plan.concern else ""
    
    # Define starter strings by persona
    if plan.persona == "Elite Match":
        starters = [
            f"An outstanding fit for the role. Brings {yoe_desc} as a {role_desc}{strength_phrase}.",
            f"Evaluated as an elite candidate, offering {yoe_desc} as a {role_desc}{strength_phrase}.",
            f"A premier profile matching all requirements, presenting {yoe_desc} at {role_desc.split(' at ')[1]}."
        ]
        sentence = starters[seed % len(starters)] + differentiator_clause
        
    elif plan.persona == "Strong Product Engineer":
        starters = [
            f"A strong product systems builder with {yoe_desc} as a {role_desc}.",
            f"Brings robust product engineering experience, offering {yoe_desc} as a {role_desc}."
        ]
        sentence = starters[seed % len(starters)] + f" They demonstrate {plan.primary_reason.lower()}{strength_phrase}.{concern_clause}"
        
    elif plan.persona == "Senior Specialist":
        starters = [
            f"A seasoned specialist bringing {yoe_desc} of deep expertise, currently {role_desc}.",
            f"Offers {yoe_desc} of senior technical experience, holding a {role_desc} role."
        ]
        sentence = starters[seed % len(starters)] + f" Highly aligned due to {plan.primary_reason.lower()}.{concern_clause}"
        
    elif plan.persona == "Leadership Profile":
        starters = [
            f"Brings valuable technical leadership experience, currently a {role_desc}.",
            f"A lead engineer with {yoe_desc} of experience, currently working at {role_desc.split(' at ')[1]}."
        ]
        sentence = starters[seed % len(starters)] + f" Positioned well due to {plan.primary_reason.lower()}{strength_phrase}.{concern_clause}"
        
    elif plan.persona == "Research-Oriented":
        starters = [
            f"A research-focused engineer with {yoe_desc} at {role_desc.split(' at ')[1]}.",
            f"Supported by strong academic/research credentials, they bring {yoe_desc} as a {role_desc}."
        ]
        sentence = starters[seed % len(starters)] + f" Demonstrates {plan.secondary_reason}.{concern_clause}"
        
    elif plan.persona == "Fast Hire":
        starters = [
            f"A highly active candidate available immediately, bringing {yoe_desc} as a {role_desc}.",
            f"Ready for fast onboarding, they offer {yoe_desc} of experience at {role_desc.split(' at ')[1]}."
        ]
        sentence = starters[seed % len(starters)] + f" Strengths include {plan.secondary_reason}.{concern_clause}"
        
    elif plan.persona == "Consulting Background":
        starters = [
            f"Brings a delivery-focused background with {yoe_desc} as a {role_desc}.",
            f"An experienced systems delivery specialist offering {yoe_desc} as a {role_desc}."
        ]
        sentence = starters[seed % len(starters)] + f" Positioned as a consulting resource with {plan.primary_reason.lower()}.{concern_clause}"
        
    elif plan.persona == "Emerging Candidate":
        starters = [
            f"An emerging technical talent bringing {yoe_desc} of experience, currently {role_desc}.",
            f"A high-potential ML engineer offering {yoe_desc} as a {role_desc}."
        ]
        sentence = starters[seed % len(starters)] + f" Highlights include {plan.secondary_reason}.{concern_clause}"
        
    elif plan.persona == "High Risk":
        starters = [
            f"A candidate holding a {role_desc} role with {yoe_desc}.",
            f"Brings {yoe_desc} of experience, currently in a {role_desc} role."
        ]
        # Always emphasize concern for high risk
        concern_str = plan.concern.lower() if plan.concern else "lower overall feature scores"
        sentence = starters[seed % len(starters)] + f" However, they present a key hiring risk due to {concern_str}."
        
    else:  # Backend Generalist or fallback
        starters = [
            f"Offers a solid generalist background with {yoe_desc} as a {role_desc}.",
            f"Brings {yoe_desc} of technical experience, currently at {role_desc.split(' at ')[1]}."
        ]
        sentence = starters[seed % len(starters)] + f" Positioned on the list with {plan.primary_reason.lower()}.{concern_clause}"
        
    # Clean up double spaces or minor punctuation anomalies
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


# Runs the 4-stage reasoning pipeline for the top candidates.
def generate_reasonings(candidates_df: pd.DataFrame) -> list[str]:
    analyses = []
    for _, row in candidates_df.iterrows():
        analyses.append(analyze_candidate(row))
        
    reasonings = []
    for idx, analysis in enumerate(analyses):
        # Pairwise differentiator comparison with the adjacent next candidate (rank i+1)
        next_analysis = analyses[idx + 1] if idx + 1 < len(analyses) else None
        
        # 1. Create Reason Plan (Core Intelligence)
        plan = plan_reason(analysis, next_analysis)
        
        # 2. Run Natural Language Generator
        summary = generate_summary_from_plan(plan)
        reasonings.append(summary)
        
    return reasonings
