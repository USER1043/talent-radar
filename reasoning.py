"""
@file Deterministic reasoning generation pipeline for candidate ranking.
@package online_ranking
"""

from __future__ import annotations

import json
import re
import hashlib
import torch
import pandas as pd
from dataclasses import dataclass

_MODEL = None
_TOKENIZER = None


# Loads the local Qwen model and tokenizer lazily.
def _get_model_and_tokenizer():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    return _MODEL, _TOKENIZER


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
    skills: list[str]


@dataclass
class DecisionAnalysis:
    """Decision details analyzing why a candidate is placed at their rank."""
    primary_reason: str
    strongest_positive: str
    strongest_negative: str
    why_not_higher: str
    why_not_lower: str
    risk: str | None
    recommendation: str
    coverage_percent: float
    missing_requirements: list[str]


@dataclass
class ReasonPlan:
    """Recruiter reasoning plan containing planned drivers and differentiator."""
    candidate_id: str
    persona: str
    decision: DecisionAnalysis
    evidence: list[str]
    differentiator_beats: str | None
    differentiator_loses: str | None
    rank: int


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
    
    try:
        skills = [s["name"].lower() for s in json.loads(row["skills_json"])]
    except (json.JSONDecodeError, TypeError, KeyError):
        skills = []
        
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
        skills=skills,
    )


# Evaluates candidate feature criteria to produce structured decision metrics.
def perform_decision_analysis(analysis: CandidateAnalysis, rank: int) -> DecisionAnalysis:
    # Compute explicit requirement coverage
    jd_reqs = ["retrieval", "vector", "python", "product ml"]
    matched_reqs = []
    
    skills_concat = " ".join(analysis.skills).lower()
    if any(x in skills_concat for x in ["retrieval", "search", "bm25", "elasticsearch"]):
        matched_reqs.append("retrieval")
    if any(x in skills_concat for x in ["vector", "faiss", "qdrant", "weaviate", "milvus", "pinecone"]):
        matched_reqs.append("vector")
    if "python" in skills_concat:
        matched_reqs.append("python")
    if not analysis.consulting_only and analysis.yoe >= 3.0:
        matched_reqs.append("product ml")
        
    missing_reqs = [r for r in jd_reqs if r not in matched_reqs]
    coverage_percent = len(matched_reqs) / len(jd_reqs)

    # Strongest positive signal
    if analysis.semantic_match >= 0.58:
        pos = "Exceptional semantic match to ideal description"
    elif not analysis.consulting_only and analysis.yoe >= 5.0:
        pos = "Proven product engineering tenure"
    elif analysis.has_recent_coding:
        pos = "Active hands-on coding focus"
    else:
        pos = "Relevant industry background"

    # Strongest negative signal
    if not analysis.has_recent_coding:
        neg = "Absence of recent hands-on coding"
    elif analysis.notice_days > 45:
        neg = f"{int(analysis.notice_days)}-day notice period constraint"
    elif analysis.consulting_only:
        neg = "Consulting-only history"
    elif analysis.semantic_match < 0.54:
        neg = "Lower semantic relevance to candidate profile specifications"
    else:
        neg = "Absence of strong external validation markers"

    # Primary Reason
    if rank <= 10:
        primary = "exceptional overall alignment across semantic, technical, and readiness dimensions"
    elif analysis.semantic_match >= 0.56:
        primary = "strong search and retrieval core skills alignment"
    elif not analysis.consulting_only:
        primary = "relevant product systems background"
    else:
        primary = "delivery-focused engineering skills background"

    # Why not ranked higher
    if not analysis.has_recent_coding:
        why_higher = "their current role is assessed as non-coding/management"
    elif analysis.notice_days > 30:
        why_higher = f"their long {int(analysis.notice_days)}-day notice period introduces timeline risk"
    elif analysis.semantic_match < 0.56:
        why_higher = "adjacent candidates show closer semantic similarity to search constraints"
    elif not analysis.open_source and not analysis.publications:
        why_higher = "they lack external public technical indicators (OSS/publications)"
    else:
        why_higher = "higher-ranked candidates present stronger behavioral activity scores"

    # Why not ranked lower
    if analysis.semantic_match >= 0.55:
        why_lower = "their search architecture similarity remains strong"
    elif analysis.yoe >= 5.0:
        why_lower = "their overall professional experience anchors the profile value"
    elif analysis.has_recent_coding:
        why_lower = "they maintain active technical contributions"
    else:
        why_lower = "their active platform indicators verify readiness"

    # Hiring Risk
    risk = None
    if analysis.notice_days > 60:
        risk = "Hiring timeline constraint"
    elif not analysis.has_recent_coding:
        risk = "Technical skill decay"
    elif analysis.consulting_only:
        risk = "Delivery-oriented culture shift"

    rec = "Shortlist" if rank <= 30 else "Evaluate with reservation"

    return DecisionAnalysis(
        primary_reason=primary,
        strongest_positive=pos,
        strongest_negative=neg,
        why_not_higher=why_higher,
        why_not_lower=why_lower,
        risk=risk,
        recommendation=rec,
        coverage_percent=coverage_percent,
        missing_requirements=missing_reqs,
    )


# Formulates the structural reason plan including differentiators and personas.
def plan_reason(
    analysis: CandidateAnalysis,
    prev_analysis: CandidateAnalysis | None,
    next_analysis: CandidateAnalysis | None,
    rank: int
) -> ReasonPlan:
    # 1. Persona Classification
    if analysis.yoe >= 5.0 and analysis.semantic_match >= 0.58 and analysis.behavior_score >= 0.70 and analysis.notice_days <= 30:
        persona = "Elite Match"
    elif not analysis.has_recent_coding or analysis.notice_days > 90 or (analysis.semantic_match < 0.52 and analysis.notice_days > 60):
        persona = "High Risk"
    elif analysis.yoe >= 4.0 and not analysis.consulting_only and analysis.semantic_match >= 0.55 and analysis.has_recent_coding:
        persona = "Strong Product Engineer"
    elif analysis.yoe >= 7.0 and analysis.semantic_match >= 0.55:
        persona = "Senior Specialist"
    elif analysis.leadership and analysis.yoe >= 5.0:
        persona = "Leadership Profile"
    elif analysis.publications:
        persona = "Research-Oriented"
    elif analysis.notice_days <= 15:
        persona = "Fast Hire"
    elif analysis.consulting_only:
        persona = "Consulting Background"
    elif analysis.yoe < 4.0 and analysis.semantic_match >= 0.53:
        persona = "Emerging Candidate"
    else:
        persona = "Backend Generalist"

    # 2. Decision Analysis
    decision = perform_decision_analysis(analysis, rank)

    # 3. Evidence list prioritisation
    evidence = []
    if analysis.semantic_match >= 0.55:
        evidence.append("strong search alignment")
    if not analysis.consulting_only and analysis.yoe >= 3.0:
        evidence.append("product experience")
    if analysis.notice_days <= 30:
        evidence.append("favorable notice period")
    if analysis.open_source:
        evidence.append("open source validation")
    if analysis.publications:
        evidence.append("academic research profile")
    
    # Ensure we limit evidence items to the most crucial 2
    evidence = evidence[:2] if evidence else ["applied engineering experience"]

    # 4. Pairwise Differentiators (Why beats next, why loses to prev)
    diff_beats = None
    if next_analysis is not None:
        if analysis.notice_days < next_analysis.notice_days - 15:
            diff_beats = "a shorter notice period timeline"
        elif analysis.semantic_match > next_analysis.semantic_match + 0.02:
            diff_beats = "stronger core search similarity"
        elif analysis.yoe > next_analysis.yoe + 2.0:
            diff_beats = "additional years of technical experience"
        elif analysis.has_recent_coding and not next_analysis.has_recent_coding:
            diff_beats = "active coding responsibilities"

    diff_loses = None
    if prev_analysis is not None:
        if analysis.notice_days > prev_analysis.notice_days + 15:
            diff_loses = "longer notice period timeline"
        elif analysis.semantic_match < prev_analysis.semantic_match - 0.02:
            diff_loses = "weaker semantic similarity to the ideal JD description"
        elif analysis.yoe < prev_analysis.yoe - 2.0:
            diff_loses = "fewer years of senior-level experience"
        elif not analysis.has_recent_coding and prev_analysis.has_recent_coding:
            diff_loses = "lack of recent hands-on coding"

    return ReasonPlan(
        candidate_id=analysis.candidate_id,
        persona=persona,
        decision=decision,
        evidence=evidence,
        differentiator_beats=diff_beats,
        differentiator_loses=diff_loses,
        rank=rank,
    )


# Helper helper function for checking evidence text names.
def summary_comp_check(evidence: list[str]) -> str:
    return " ".join(evidence).lower()


# Validates that the generated summary statement has no logical contradictions or pronoun leaks.
def validate_explanation(summary: str, plan: ReasonPlan) -> str:
    # Leak Checks
    first_person_markers = ["i ", "my ", "we ", "our ", "us ", "me "]
    for marker in first_person_markers:
        if marker in summary.lower():
            summary = re.sub(r"\b(my|our|us|me)\b", "the", summary, flags=re.IGNORECASE)
            summary = re.sub(r"\b(i)\b", "candidate", summary, flags=re.IGNORECASE)

    # Logic self-correction rules to prevent invalid claims
    if plan.rank <= 10:
        summary = summary.replace("is caution-flagged", "presents a minor notice constraint")
        summary = summary.replace("evaluated as a high risk candidate", "has a structured background")
        
    if "minimal gap" in summary.lower() and plan.rank > 50:
        summary = summary.replace("minimal gap", "specific technical limitations")
        
    return summary


# Fallback template variation generator in case LLM fails.
def fallback_summary_generator(plan: ReasonPlan) -> str:
    dec = plan.decision
    evidence_str = " and ".join(plan.evidence)
    seed = int(hashlib.md5(plan.candidate_id.encode()).hexdigest(), 16) % 100
    
    if plan.rank <= 10:
        starters = [
            f"Ranked #{plan.rank} because the profile combines {dec.strongest_positive.lower()}, recent hands-on engineering work, and {evidence_str}. No significant hiring constraints prevented placement at the top of the list.",
            f"Placed at #{plan.rank} as the candidate profile combines {dec.strongest_positive.lower()} with recent hands-on work and {evidence_str}. No major pipeline constraints affected top ranking.",
            f"Ranked #{plan.rank} matching the top-tier requirement profile. Combines {dec.strongest_positive.lower()} and recent hands-on work with {evidence_str} without any notable hiring constraints."
        ]
        summary = starters[seed % len(starters)]
        if plan.differentiator_beats:
            summary += f" Stands out from adjacent profiles by offering {plan.differentiator_beats}."
        return validate_explanation(summary, plan)

    imposing_companies = ["google", "apple", "microsoft", "amazon", "netflix", "adobe", "meta", "salesforce"]
    is_big_tech = False
    for comp in imposing_companies:
        if comp in dec.strongest_positive.lower() or comp in dec.strongest_negative.lower():
            is_big_tech = True

    if plan.rank > 20 and is_big_tech:
        starters = [
            f"Although the candidate offers a pedigree profile with experience matching {dec.strongest_positive.lower()}, they are positioned lower at #{plan.rank} because {dec.why_not_higher}. Placement is anchored here as they maintain {dec.why_not_lower}.",
            f"While the candidate brings pedigree experience matching {dec.strongest_positive.lower()}, the profile is ranked at #{plan.rank} because {dec.why_not_higher}. Upward progression is capped, though their position is anchored by {dec.why_not_lower}.",
            f"Offers a tech pedigree profile with experience matching {dec.strongest_positive.lower()}, but ranks lower at #{plan.rank} because {dec.why_not_higher}. Secured here due to {dec.why_not_lower}."
        ]
        summary = starters[seed % len(starters)]
        return validate_explanation(summary, plan)

    if plan.persona == "High Risk":
        starters = [
            f"Ranked #{plan.rank} primarily due to {dec.strongest_negative.lower()}, which acts as a primary constraint. While they present {evidence_str}, {dec.why_not_higher}, keeping them below top-tier candidates.",
            f"Placement at #{plan.rank} is restricted due to {dec.strongest_negative.lower()}; this constraint keeps them below the top list. Although they demonstrate {evidence_str}, {dec.why_not_higher}.",
            f"Positioned at #{plan.rank} because the profile presents a key limitation of {dec.strongest_negative.lower()}. While they bring {evidence_str}, upward movement is limited because {dec.why_not_higher}."
        ]
        summary = starters[seed % len(starters)]
    elif plan.persona == "Strong Product Engineer":
        starters = [
            f"Positioned at #{plan.rank} representing a solid candidate with {evidence_str}. They do not rank higher because {dec.why_not_higher}; however, they remain anchored here due to {dec.why_not_lower}.",
            f"Placed at #{plan.rank} as a strong product engineer offering {evidence_str}. Upward ranking is constrained as {dec.why_not_higher}, but they maintain this position because {dec.why_not_lower}.",
            f"Brings a robust profile at #{plan.rank} highlighted by {evidence_str}. While {dec.why_not_higher} restricts them from higher slots, they avoid falling lower due to {dec.why_not_lower}."
        ]
        summary = starters[seed % len(starters)]
    elif plan.persona == "Senior Specialist":
        starters = [
            f"Placed at #{plan.rank} offering senior-level specialist background. Further upward placement is limited as {dec.why_not_higher}, but they beat lower profiles due to {dec.why_not_lower}.",
            f"Offers senior specialist expertise at #{plan.rank}. They are kept out of higher tiers because {dec.why_not_higher}, though their position is secured by {dec.why_not_lower}.",
            f"Ranked #{plan.rank} with a focus on senior technical specialization. The profile is limited because {dec.why_not_higher}, but maintains value due to {dec.why_not_lower}."
        ]
        summary = starters[seed % len(starters)]
    elif plan.persona == "Elite Match":
        starters = [
            f"Ranked at #{plan.rank} demonstrating strong JD compatibility. Timeline constraints like {dec.strongest_negative.lower()} restrict them from the top 10 positions.",
            f"Holds rank #{plan.rank} as a high-compatibility profile. However, {dec.strongest_negative.lower()} prevents them from placing higher on the list.",
            f"Brings high-quality alignment at #{plan.rank}, though {dec.strongest_negative.lower()} keeps them outside the top-tier rankings."
        ]
        summary = starters[seed % len(starters)]
    else:  # General trade-off template
        starters = [
            f"Ranked at #{plan.rank} based on {dec.primary_reason}. Brings {evidence_str}, but {dec.why_not_higher}. They maintain position ahead of lower candidates because {dec.why_not_lower}.",
            f"Placed at #{plan.rank} due to {dec.primary_reason}. The profile offers {evidence_str}, but is limited because {dec.why_not_higher}. Position is secured because {dec.why_not_lower}.",
            f"Positioned at #{plan.rank} showing {dec.primary_reason}. While they offer {evidence_str}, {dec.why_not_higher}. They avoid lower placement because {dec.why_not_lower}."
        ]
        summary = starters[seed % len(starters)]

    if plan.differentiator_beats and plan.rank % 2 == 0:
        summary += f" They stand out from lower-ranked profiles via {plan.differentiator_beats}."
        
    return validate_explanation(summary, plan)


# Runs the decision-analysis and reasoning generation pipeline with batched guided Qwen execution.
def generate_reasonings(candidates_df: pd.DataFrame) -> list[str]:
    analyses = []
    for _, row in candidates_df.iterrows():
        analyses.append(analyze_candidate(row))
        
    plans = []
    for idx, analysis in enumerate(analyses):
        prev_analysis = analyses[idx - 1] if idx - 1 >= 0 else None
        next_analysis = analyses[idx + 1] if idx + 1 < len(analyses) else None
        rank = idx + 1
        plans.append(plan_reason(analysis, prev_analysis, next_analysis, rank))

    try:
        torch.set_num_threads(8)
        model, tokenizer = _get_model_and_tokenizer()
        
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        prompts = []
        for plan in plans:
            dec = plan.decision
            evidence_str = ", ".join(plan.evidence)
            
            facts = (
                f"- Candidate Rank: #{plan.rank} of 100\n"
                f"- Candidate Persona: {plan.persona}\n"
                f"- Primary Strength: {dec.primary_reason}\n"
                f"- Supporting Evidence: {evidence_str}\n"
                f"- Hiring Constraint: {dec.strongest_negative if dec.strongest_negative else 'none'}\n"
                f"- Limit reason (why not higher): {dec.why_not_higher}\n"
                f"- Differentiator: {plan.differentiator_beats if plan.differentiator_beats else 'none'}"
            )
            
            messages = [
                {"role": "system", "content": (
                    "You are a professional recruiting coordinator. Write a brief, objective 1-2 sentence candidate summary "
                    "for a hiring manager using only the facts below. Your explanation MUST justify why this candidate is at "
                    "this exact rank and describe the trade-offs (e.g. why not higher). Do NOT use first-person pronouns (I, my, our, we).\n\n"
                    "Few-Shot Examples:\n\n"
                    "Facts:\n"
                    "- Candidate Rank: #12 of 100\n"
                    "- Candidate Persona: Senior Specialist\n"
                    "- Primary Strength: strong search and retrieval core skills alignment\n"
                    "- Supporting Evidence: strong search alignment, product experience\n"
                    "- Hiring Constraint: 90-day notice period constraint\n"
                    "- Limit reason (why not higher): their long 90-day notice period introduces timeline risk\n"
                    "- Differentiator: shorter notice period timeline\n"
                    "Summary:\n"
                    "Placed at #12, the candidate offers senior specialist depth with strong search alignment. They are kept out of higher tiers because their long 90-day notice period introduces timeline risk, but they beat lower profiles due to their product experience.\n\n"
                    "Facts:\n"
                    "- Candidate Rank: #98 of 100\n"
                    "- Candidate Persona: High Risk\n"
                    "- Primary Strength: relevant product systems background\n"
                    "- Supporting Evidence: product experience\n"
                    "- Hiring Constraint: 120-day notice period constraint\n"
                    "- Limit reason (why not higher): their long 120-day notice period introduces timeline risk\n"
                    "- Differentiator: none\n"
                    "Summary:\n"
                    "Ranked #98 primarily due to their 120-day notice period constraint, which acts as a primary timeline risk. While they present product experience, their long notice period limits their ranking and keeps them below top-tier candidates."
                )},
                {"role": "user", "content": f"Facts:\n{facts}\n\nSummary:"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            
        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=85,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        reasonings = []
        for idx, plan in enumerate(plans):
            input_len = len(inputs.input_ids[idx])
            output_ids = generated_ids[idx][input_len:]
            response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            response = response.replace("\n", " ").strip()
            
            if not response.endswith((".", "!", "?")):
                matches = list(re.finditer(r'[.!?](?:\s|$)', response))
                if matches:
                    last_punc_idx = matches[-1].start()
                    response = response[:last_punc_idx + 1]
                else:
                    response = response + "."
            
            reasonings.append(validate_explanation(response, plan))
            
        return reasonings

    except Exception as e:
        print(f"LLM Batch generation failed: {e}. Falling back to multi-template.")
        return [fallback_summary_generator(p) for p in plans]
