"""
@file Generates grounded reasoning statements for the top-ranked candidates using a local LLM.
@package online_ranking
"""

from __future__ import annotations

import json
import re
import torch
import pandas as pd
import difflib
from pathlib import Path

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


# Checks if the response contains any logical contradictions or incorrect comparisons for notice days.
def has_notice_hallucination(resp: str, days: float) -> bool:
    if pd.isna(days):
        return False
    days_int = int(days)
    resp_lower = resp.lower()
    
    # 1. Catch "preferred/preference of X" where X > 30
    if days > 30:
        bad_pref_patterns = [
            f"preferred {days_int}",
            f"preference of {days_int}",
            f"preferred notice of {days_int}",
            f"preferred notice period of {days_int}",
            f"{days_int}-day preference",
            f"{days_int} day preference",
            f"preference is {days_int}",
            f"buyout of {days_int}",
            f"buy out of {days_int}"
        ]
        for p in bad_pref_patterns:
            if p in resp_lower:
                return True
                
    # 2. Catch comparative self-contradictions (e.g. "more than X", "less than X", "longer than X", "shorter than X")
    bad_comp_patterns = [
        f"more than {days_int}",
        f"less than {days_int}",
        f"longer than {days_int}",
        f"shorter than {days_int}",
        f"exceeds {days_int}"
    ]
    for p in bad_comp_patterns:
        if p in resp_lower:
            return True
            
    return False


# Verifies that any numbers appearing in the response are grounded in the allowed numbers.
def verify_number_grounding(resp: str, allowed_numbers: list[str]) -> bool:
    found_numbers = re.findall(r"\b\d+\b", resp)
    for num in found_numbers:
        if num not in allowed_numbers:
            return False
    return True


# Checks if the generated text is strictly grounded in the allowed facts.
def verify_grounding(text: str, allowed_skills: list[str], allowed_companies: list[str]) -> bool:
    text_lower = text.lower()
    # Split text into a set of whole words for safe word-boundary matching
    text_words = set(re.findall(r"[a-z0-9]+", text_lower))

    def _word_set(s: str) -> set[str]:
        """Returns the set of alphanumeric tokens in a string."""
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    # Precompute word sets for all allowed entities once
    allowed_skill_wordsets = [_word_set(s) for s in allowed_skills]
    allowed_co_wordsets = [_word_set(c) for c in allowed_companies]

    # List of all known skills in the dataset
    all_known_skills = [
        "langchain", "llamaindex", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "elasticsearch", "faiss", "python", "pytorch", "tensorflow",
        "scikit-learn", "numpy", "pandas", "lora", "qlora", "peft", "xgboost",
        "nlp", "computer vision", "opencv", "cnn", "yolo", "gans", "diffusion models"
    ]

    for s in all_known_skills:
        s_words = _word_set(s)
        # Only flag if ALL words of the known skill token appear in the text
        if s_words.issubset(text_words):
            # Pass if any allowed skill shares all those words (whole-word overlap)
            if not any(s_words.issubset(aws) or aws.issubset(s_words) for aws in allowed_skill_wordsets):
                return False

    # List of known companies
    known_cos = [
        "apple", "amazon", "ola", "sarvam", "phonepe", "meta", "adobe", "google",
        "tcs", "wipro", "infosys", "mindtree", "cognizant", "capgemini", "accenture",
        "acme", "dunder mifflin", "globex", "initech", "wayne enterprises", "hooli",
        "stark industries", "pied piper", "niramai", "locobuzz", "paytm", "unacademy",
        "byju's", "vedantu", "flipkart", "swiggy", "zomato", "freshworks", "haptik",
        "observe.ai", "saarthi.ai", "rephrase.ai", "razorpay", "meesho", "dream11"
    ]

    for c in known_cos:
        c_words = _word_set(c)
        # Only flag if ALL words of the known company token appear in the text
        if c_words.issubset(text_words):
            # Pass if any allowed company shares at least one word with the token
            if not any(c_words & acw for acw in allowed_co_wordsets):
                return False

    return True


# Maps specific skill names to broad, grounding-safe category labels.
def _skill_to_category(skill: str) -> str:
    s = skill.lower()
    if any(x in s for x in ["vector", "faiss", "pinecone", "qdrant", "weaviate", "milvus", "opensearch", "elasticsearch"]):
        return "vector/search infrastructure"
    if any(x in s for x in ["embeddings", "sentence-transformer", "semantic"]):
        return "embedding systems"
    if any(x in s for x in ["lora", "qlora", "peft", "fine-tun", "finetun"]):
        return "LLM fine-tuning"
    if any(x in s for x in ["ranking", "ndcg", "learning to rank", "lambdamart", "xgboost", "lgbm"]):
        return "learning-to-rank systems"
    if any(x in s for x in ["nlp", "transformers", "bert", "gpt"]):
        return "NLP / transformer models"
    if any(x in s for x in ["recommendation", "recsys"]):
        return "recommendation systems"
    if any(x in s for x in ["retrieval", "bm25", "hybrid search"]):
        return "information retrieval"
    if any(x in s for x in ["mlops", "deployment", "serving", "triton", "kubernetes", "docker"]):
        return "ML deployment / MLOps"
    return "applied ML"


# Checks if the generated text contains any first-person markers.
def is_first_person(text: str) -> bool:
    first_person_markers = {"i am", "i have", "my experience", "i've", "i work", "myself", "my background"}
    text_lower = text.lower()
    return any(marker in text_lower for marker in first_person_markers)


# Checks if the generated text contains any non-English characters.
def has_non_english(text: str) -> bool:
    for c in text:
        o = ord(c)
        if o > 127:
            # Allow common unicode punctuation/dashes/quotes (general punctuation range 0x2000-0x206F)
            if not (0x2000 <= o <= 0x206F):
                return True
    return False


# Checks if the reasoning text contains concern or hedging language for rank > 20.
def has_concern_hedge(text: str, rank: int, gap: str) -> bool:
    if rank <= 20:
        return True
    text_lower = text.lower()
    hedging_markers = {
        "but", "however", "lacks", "lack", "does not yet"
    }
    return any(h in text_lower for h in hedging_markers)


# Extracts the hedging clause from the generated candidate reasoning text.
def extract_hedge_clause(text: str) -> str:
    text_lower = text.lower()
    splitters = ["but", "however", "does not yet", "lacks", "though", "yet"]
    for splitter in splitters:
        if splitter in text_lower:
            return text_lower.split(splitter, 1)[1].strip()
    return text_lower


# Computes the sequence matcher similarity ratio between two strings.
def get_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


JD_REQUIREMENT_LABELS = {
    "semantic_sim_to_ideal": "production-scale search/retrieval experience",
    "semantic_sim_to_jd": "NLP/retrieval systems expertise",
    "years_of_experience": "senior-level professional experience",
    "no_recent_coding": "recent hands-on coding",
    "is_consulting_only": "product company experience",
    "is_pure_research": "shipped production ML systems",
    "no_external_validation": "external validation (OSS/publications)",
    "cv_speech_robotics_no_nlp": "NLP/information retrieval background",
    "is_recent_wrapper_only": "pre-LLM ML/AI core foundation",
    "behavioral_score": "recruiter response and platform activity",
    "notice_period_penalty": "JD-preferred notice period",
    "bm25_score": "keyword/search systems expertise",
}


# Loads the ranker model coefficients to determine feature contributions.
def _get_ranker_coefs() -> dict[str, float]:
    model_path = Path("artifacts/ranker_model.pkl")
    if model_path.exists():
        try:
            import pickle
            with open(model_path, "rb") as f:
                payload = pickle.load(f)
            model = payload["model"]
            feature_names = payload["feature_names"]
            if hasattr(model, "coef_"):
                return dict(zip(feature_names, model.coef_))
        except Exception:
            pass
    return {
        'years_of_experience': -0.17003386, 'n_skills': 0.08163438, 'is_consulting_only': -0.00333992,
        'is_pure_research': 0.0, 'is_recent_wrapper_only': 0.0, 'no_recent_coding': -0.75688456,
        'cv_speech_robotics_no_nlp': -0.24919117, 'no_external_validation': -0.6123869,
        'title_chasing_pattern': 0.0405579, 'framework_enthusiast_signal': -0.22621816,
        'behavioral_score': 1.59216212, 'notice_period_penalty': 0.16015115,
        'skill_to_experience_ratio': -0.21268627, 'semantic_sim_to_jd': -0.86485791,
        'semantic_sim_to_ideal': 2.396714, 'bm25_score': 0.00388941
    }


def generate_looser_reasoning(
    model, tokenizer,
    yoe: float, title: str, company: str, rank: int, gap: str,
    matched_skills: list[str], past_roles: str, jd_label: str
) -> str:
    # Convert specific skill names to safe category labels
    skill_categories = list(dict.fromkeys(_skill_to_category(s) for s in matched_skills[:4]))
    skills_summary = ", ".join(skill_categories[:2]) if skill_categories else "applied ML systems"

    if rank <= 20:
        facts = (
            f"- Current role: {title} at {company}\n"
            f"- Skill areas: {skills_summary}\n"
            f"- Past roles: {past_roles}\n"
            f"- Rank: #{rank} of 100\n"
            f"- Strongest match to role: {jd_label}"
        )
        messages = [
            {"role": "system", "content": (
                "You are a professional recruiting assistant. Write exactly ONE complete sentence (under 18 words) "
                "summarising this candidate's fit for a Senior AI/ML Engineer role, using the facts below. "
                "GOOD example: '7 years at Google building large-scale retrieval, directly matching the role's need for production search systems.' "
                "BAD example: '7 years of experience at Google, skilled in FAISS.' "
                "Reference the strongest match to the role in your sentence to show how it connects to the JD requirement. "
                "Do NOT copy skill tool names verbatim. Do NOT mention any company not listed below."
            )},
            {"role": "user", "content": f"Facts:\n{facts}\n\nOutput one complete sentence ending with a period."}
        ]
    else:
        facts = (
            f"- Current role: {title} at {company}\n"
            f"- Skill areas: {skills_summary}\n"
            f"- Past roles: {past_roles}\n"
            f"- Rank: #{rank} of 100\n"
            f"- Concern: {gap}\n"
            f"- JD requirement gap: {jd_label}"
        )
        messages = [
            {"role": "system", "content": (
                "You are a professional recruiting assistant. Write exactly ONE complete sentence (under 22 words) "
                "summarising this candidate's fit for a Senior AI/ML Engineer role, weaving in a description of the "
                "specific concern below in your own words — do NOT copy the concern phrase verbatim. "
                "GOOD example: '7 years at Google building large-scale retrieval, directly matching the role's need for production search systems.' "
                "BAD example: '7 years of experience at Google, skilled in FAISS.' "
                "Connect the concern to the JD requirement gap in your sentence. "
                "Do NOT copy skill tool names verbatim. Do NOT mention any company not listed below."
            )},
            {"role": "user", "content": f"Facts:\n{facts}\n\nRules:\n1. Output exactly one complete sentence ending with a period.\n2. You MUST explicitly state the concern and connect it to the JD requirement gap. Do NOT write a positive-only sentence."}
        ]
        
    with torch.no_grad():
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
    if not response.endswith((".", "!", "?")):
        matches = list(re.finditer(r'[.!?](?:\s|$)', response))
        if matches:
            last_punc_idx = matches[-1].start()
            response = response[:last_punc_idx + 1]
        else:
            response = response + "."
            



# Generates reasoning statements for the final 100 candidates.
def generate_reasonings(candidates_df: pd.DataFrame) -> list[str]:
    torch.set_num_threads(8)
    model, tokenizer = _get_model_and_tokenizer()
    
    # Configure padding for batched generation
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define JD key skills
    jd_skills = {
        "sentence-transformers", "embeddings", "vector search", "hybrid search", 
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", 
        "faiss", "python", "ndcg", "mrr", "map", "a/b testing", "lora", "qlora", 
        "peft", "learning to rank", "xgboost", "neural ranking", "hr-tech", 
        "distributed systems", "nlp", "information retrieval"
    }

    # Pre-parse and set up candidate contexts
    candidates = []
    for idx, (_, row) in enumerate(candidates_df.iterrows()):
        yoe = row["years_of_experience"]
        title = row["current_title"]
        company = row["current_company"]
        rank = int(row["rank"])
        
        # Parse skills
        try:
            skills_list = [s["name"] for s in json.loads(row["skills_json"])]
        except (json.JSONDecodeError, TypeError, KeyError):
            skills_list = []
            
        # Add domain-specific skills found in job titles to the allowed list to prevent title-based grounding failures
        title_skills = []
        all_known_skills = [
            "langchain", "llamaindex", "pinecone", "weaviate", "qdrant", "milvus",
            "opensearch", "elasticsearch", "faiss", "python", "pytorch", "tensorflow",
            "scikit-learn", "numpy", "pandas", "lora", "qlora", "peft", "xgboost",
            "nlp", "computer vision", "opencv", "cnn", "yolo", "gans", "diffusion models"
        ]
        
        # Parse career
        try:
            career_list = json.loads(row["career_history_json"])
        except (json.JSONDecodeError, TypeError):
            career_list = []

        for t in [title] + [c.get("title") for c in career_list if c.get("title")]:
            t_lower = t.lower()
            for s in list(jd_skills) + all_known_skills:
                if s.lower() in t_lower:
                    title_skills.append(s)
        skills_list.extend(title_skills)
        
        matched_skills = [s for s in skills_list if s.lower() in jd_skills or any(x in s.lower() for x in ["embeddings", "vector", "search", "ranking", "eval", "llm", "finetun", "fine-tun", "nlp", "retrieval"])]
        if not matched_skills:
            matched_skills = skills_list[:5]
            
        roles = []
        companies = [company]
        for c in career_list[:2]:
            roles.append(f"{c.get('title')} at {c.get('company')} ({c.get('duration_months', 0)} months)")
            if c.get("company"):
                companies.append(c.get("company"))
        relevant_career = "; ".join(roles)
        days = row["redrob_notice_period_days"]
        
        # 1. Define fully resolved descriptions for all features (hyper-specific to Senior AI Engineer JD)
        JD_REQUIREMENT_DESCRIPTIONS = {
            "semantic_sim_to_ideal": {
                "strength": "strong alignment with the ideal senior AI/search engineer archetype (applied ML, search/retrieval at product companies)",
                "gap": "profile narrative does not closely match the ideal senior AI/search engineer archetype (applied ML, search/retrieval at product companies)"
            },
            "semantic_sim_to_jd": {
                "strength": "excellent alignment with the JD's NLP and retrieval systems requirements",
                "gap": "limited specific alignment with the JD's NLP and retrieval systems requirements"
            },
            "years_of_experience": {
                "strength": lambda val: f"senior-level professional experience of {val:.1f} years, matching the JD's target of 6-8 years total experience (with 4-5 years in applied ML/AI roles)",
                "gap": lambda val: f"only {val:.1f} years of experience, below the JD's target of 6-8 years total experience" if val < 6.0 else f"{val:.1f} years of experience without matching seniority markers (open-source / publications)"
            },
            "no_recent_coding": {
                "strength": "active hands-on coding role in recent months",
                "gap": "has not written production code in the last 18 months"
            },
            "is_consulting_only": {
                "strength": "valuable product company experience",
                "gap": "career has been entirely at consulting firms with no product company experience"
            },
            "is_pure_research": {
                "strength": "strong experience shipping production systems",
                "gap": "comes from a pure research background with no shipped production systems"
            },
            "no_external_validation": {
                "strength": "external validation via open-source projects, talks, or publications",
                "gap": "lacks external validation via open-source projects, talks, or publications"
            },
            "cv_speech_robotics_no_nlp": {
                "strength": "strong NLP and information retrieval background",
                "gap": "background is in computer vision, speech, or robotics without significant NLP or IR exposure"
            },
            "is_recent_wrapper_only": {
                "strength": "substantial pre-LLM-era ML production experience and a solid core ML/AI foundation",
                "gap": "recent AI experience consists primarily of wrapper projects using API calls (using LangChain) under 12 months, lacking substantial pre-LLM ML/AI core foundation"
            },
            "behavioral_score": {
                "strength": "high platform activity and recruiter response rate",
                "gap": "low recent platform activity and recruiter response rate on the platform"
            },
            "notice_period_penalty": {
                "strength": lambda val: "short notice period (sub-30-day notice, no buyout required)",
                "gap": lambda val: f"notice period of {int(val)} days is longer than the JD's preferred sub-30-day notice"
            },
            "bm25_score": {
                "strength": "strong keywords matching search systems expertise",
                "gap": "limited keyword alignment with search systems expertise"
            }
        }

        # Gap logic mapping to JD label
        gap = None
        jd_label = None
        if row.get("no_recent_coding") == 1.0:
            gap = JD_REQUIREMENT_DESCRIPTIONS["no_recent_coding"]["gap"]
            jd_label = JD_REQUIREMENT_LABELS["no_recent_coding"]
        elif row.get("is_consulting_only") == 1.0:
            gap = JD_REQUIREMENT_DESCRIPTIONS["is_consulting_only"]["gap"]
            jd_label = JD_REQUIREMENT_LABELS["is_consulting_only"]
        elif row.get("cv_speech_robotics_no_nlp") == 1.0:
            gap = JD_REQUIREMENT_DESCRIPTIONS["cv_speech_robotics_no_nlp"]["gap"]
            jd_label = JD_REQUIREMENT_LABELS["cv_speech_robotics_no_nlp"]
        elif row.get("is_pure_research") == 1.0:
            gap = JD_REQUIREMENT_DESCRIPTIONS["is_pure_research"]["gap"]
            jd_label = JD_REQUIREMENT_LABELS["is_pure_research"]
        elif row.get("notice_period_penalty", 0.0) > 0.0 and pd.notna(days) and days > 30:
            gap = JD_REQUIREMENT_DESCRIPTIONS["notice_period_penalty"]["gap"](days)
            jd_label = JD_REQUIREMENT_LABELS["notice_period_penalty"]
        
        if gap is None:
            sem_ideal = float(row.get("semantic_sim_to_ideal", 1.0))
            beh = float(row.get("behavioral_score", 1.0))
            yoe_val = float(yoe)
            if yoe_val < 6.0:
                gap = JD_REQUIREMENT_DESCRIPTIONS["years_of_experience"]["gap"](yoe_val)
                jd_label = JD_REQUIREMENT_LABELS["years_of_experience"]
            elif yoe_val > 9.0:
                gap = JD_REQUIREMENT_DESCRIPTIONS["years_of_experience"]["gap"](yoe_val)
                jd_label = JD_REQUIREMENT_LABELS["years_of_experience"]
            elif sem_ideal < 0.45:
                gap = JD_REQUIREMENT_DESCRIPTIONS["semantic_sim_to_ideal"]["gap"]
                jd_label = JD_REQUIREMENT_LABELS["semantic_sim_to_ideal"]
            elif beh < 0.45:
                gap = JD_REQUIREMENT_DESCRIPTIONS["behavioral_score"]["gap"]
                jd_label = JD_REQUIREMENT_LABELS["behavioral_score"]
            elif sem_ideal < 0.55:
                gap = "moderate alignment with the ideal candidate narrative, with room to grow into a senior scope"
                jd_label = "production-scale search/retrieval experience"
            else:
                gap = "does not yet demonstrate the end-to-end system ownership this senior role requires"
                jd_label = "production-scale search/retrieval experience"
            
        if rank <= 20:
            coefs = _get_ranker_coefs()
            contributions = {}
            for feat, coef in coefs.items():
                val = float(row.get(feat, 0.0))
                contributions[feat] = val * coef
                
            penalty_features = {"notice_period_penalty", "no_recent_coding", "is_consulting_only", "no_external_validation", "is_recent_wrapper_only", "cv_speech_robotics_no_nlp"}
            valid_feats = []
            for feat in JD_REQUIREMENT_LABELS.keys():
                if feat in contributions:
                    val = float(row.get(feat, 0.0))
                    if feat in penalty_features:
                        if val == 0.0:
                            valid_feats.append(feat)
                    else:
                        valid_feats.append(feat)
                        
            if not valid_feats:
                valid_feats = ["semantic_sim_to_ideal"]
                
            best_feat = max(valid_feats, key=lambda f: contributions[f])
            jd_label = JD_REQUIREMENT_LABELS[best_feat]
            
            desc_entry = JD_REQUIREMENT_DESCRIPTIONS[best_feat]["strength"]
            if callable(desc_entry):
                if best_feat == "years_of_experience":
                    strength_desc = desc_entry(yoe_val)
                elif best_feat == "notice_period_penalty":
                    strength_desc = desc_entry(days)
                else:
                    strength_desc = desc_entry(row.get(best_feat, 0.0))
            else:
                strength_desc = desc_entry
            
            notice_str = "sub-30-day notice, no buyout needed" if (pd.notna(days) and days <= 30) else f"{int(days)}-day notice" if pd.notna(days) else "no notice period penalty"
            facts_str = f"Facts:\n- Role: {title} at {company}\n- Skills: {', '.join(matched_skills[:3])}\n- Past: {relevant_career}\n- Notice: {notice_str}\n- Strongest match to role: {strength_desc}"
            rules_str = (
                "Rules:\n"
                "1. Output exactly one complete sentence ending with a period.\n"
                "2. Reference the strongest match to the role in your sentence to show how it connects to the JD requirement.\n"
                "3. Use ONLY the listed skills and companies above."
            )
        else:
            facts_str = f"Facts:\n- Role: {title} at {company}\n- Skills: {', '.join(matched_skills[:3])}\n- Past: {relevant_career}\n- Concern: {gap}\n- JD requirement gap: {jd_label}"
            rules_str = (
                "Rules:\n"
                f"1. MOST IMPORTANT: Your sentence MUST explicitly state the concern above and connect it to the JD requirement gap: {jd_label} using a contrast word like 'but' or 'however'. Do NOT write a positive-only sentence.\n"
                "2. Output exactly one complete sentence ending with a period.\n"
                "3. Use ONLY the listed skills and companies above."
            )
            
        allowed_numbers = ["30", "18", "12"]
        if pd.notna(days):
            allowed_numbers.append(str(int(days)))
        allowed_numbers.extend(["6", "8", "4", "5", "9"])
        yoe_val = float(yoe)
        allowed_numbers.append(f"{yoe_val:.1f}")
        allowed_numbers.append(str(int(yoe_val)))
        for c in career_list:
            if c.get("duration_months"):
                allowed_numbers.append(str(c["duration_months"]))
        cand_allowed_numbers = list(set(allowed_numbers))
        
        candidates.append({
            "idx": idx,
            "candidate_id": row["candidate_id"],
            "rank": rank,
            "yoe": yoe,
            "title": title,
            "company": company,
            "gap": gap,
            "jd_label": jd_label,
            "matched_skills": matched_skills,
            "relevant_career": relevant_career,
            "allowed_skills": skills_list,
            "allowed_cos": list(set(companies)),
            "allowed_numbers": cand_allowed_numbers,
            "days": days,
            "facts_str": facts_str,
            "rules_str": rules_str,
        })

    # Output storage
    reasonings = [None] * len(candidates)
    accepted_reasonings = []
    accepted_hedges = []
    
    # Helper to generate a batch of prompts
    def generate_batch(batch_candidates, temp, do_sample, system_prompt):
        texts = []
        for cand in batch_candidates:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{cand['facts_str']}\n\n{cand['rules_str']}"}
            ]
            texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        inputs = tokenizer(texts, padding=True, return_tensors="pt").to("cpu")
        gen_kwargs = {
            "max_new_tokens": 64,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id
        }
        if do_sample:
            gen_kwargs["temperature"] = temp
            
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                **gen_kwargs
            )
        
        responses = []
        for i in range(len(batch_candidates)):
            input_len = len(inputs.input_ids[i])
            resp_ids = generated_ids[i][input_len:]
            resp = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
            responses.append(resp)
        return responses
        
    # --- ROUND 1: Attempt 1 ---
    import time
    llm_start_time = time.time()
    print("--- Running reasoning Round 1 (Attempt 1) in batches of 50 ---")
    batch_size = 50
    round1_candidates = candidates.copy()
    round1_responses = []
    system_primary = (
        "You are a professional recruiting assistant. Summarize this candidate's fit in exactly one short, complete sentence (under 20 words) using ONLY the verified facts. Do not assume or invent anything. "
        "GOOD example: '7 years at Google building large-scale retrieval, directly matching the role's need for production search systems.' "
        "BAD example: '7 years of experience at Google, skilled in FAISS.'"
    )
    
    for i in range(0, len(round1_candidates), batch_size):
        batch = round1_candidates[i:i+batch_size]
        resps = generate_batch(batch, temp=0.1, do_sample=False, system_prompt=system_primary)
        round1_responses.extend(resps)
        print(f"  Processed Round 1: {i + len(batch)}/100")
 
    # Evaluate Round 1 in order of rank
    pending_candidates = []
    for cand, resp in zip(round1_candidates, round1_responses):
        idx = cand["idx"]
        is_grounded = verify_grounding(resp, cand["allowed_skills"], cand["allowed_cos"])
        is_first_pers = is_first_person(resp)
        has_non_eng = has_non_english(resp)
        has_hedge = has_concern_hedge(resp, cand["rank"], cand["gap"])
        
        stripped = resp.strip()
        ends_with_punc = len(stripped) > 0 and stripped[-1] in {'.', '!', '?'}
        
        hedge_clause = extract_hedge_clause(resp)
        is_too_similar = False
        if cand["rank"] > 20:
            for prev_resp in accepted_reasonings:
                if get_similarity(resp.lower(), prev_resp.lower()) > 0.70:
                    is_too_similar = True
                    break
            if not is_too_similar:
                is_notice = "notice" in hedge_clause.lower()
                thresh = 0.82 if is_notice else 0.60
                for prev_hedge in accepted_hedges:
                    if get_similarity(hedge_clause, prev_hedge) > thresh:
                        is_too_similar = True
                        break
                    
        is_num_grounded = verify_number_grounding(resp, cand["allowed_numbers"])
        is_notice_ok = not has_notice_hallucination(resp, cand["days"])
        
        if is_grounded and is_num_grounded and is_notice_ok and not is_first_pers and not has_non_eng and has_hedge and ends_with_punc and not is_too_similar:
            reasonings[idx] = resp
            accepted_reasonings.append(resp)
            accepted_hedges.append(hedge_clause)
        else:
            pending_candidates.append(cand)

    # --- ROUND 2: Attempt 2 (Retry) ---
    if pending_candidates:
        print(f"\n--- Running reasoning Round 2 (Attempt 2) for {len(pending_candidates)} pending candidates in batches ---")
        system_retry = (
            "You are a professional recruiting assistant. Summarize this candidate's fit in exactly one short, complete sentence (under 20 words). You MUST strictly use ONLY the verified facts. Do not assume or invent anything. Do not mention any companies or skills not in the list below. "
            "GOOD example: '7 years at Google building large-scale retrieval, directly matching the role's need for production search systems.' "
            "BAD example: '7 years of experience at Google, skilled in FAISS.'"
        )
        round2_responses = []
        for i in range(0, len(pending_candidates), batch_size):
            batch = pending_candidates[i:i+batch_size]
            resps = generate_batch(batch, temp=0.7, do_sample=True, system_prompt=system_retry)
            round2_responses.extend(resps)
            print(f"  Processed Round 2: {i + len(batch)}/{len(pending_candidates)}")

        # Evaluate Round 2 in order of rank
        still_pending = []
        for cand, resp in zip(pending_candidates, round2_responses):
            idx = cand["idx"]
            is_grounded = verify_grounding(resp, cand["allowed_skills"], cand["allowed_cos"])
            is_first_pers = is_first_person(resp)
            has_non_eng = has_non_english(resp)
            has_hedge = has_concern_hedge(resp, cand["rank"], cand["gap"])
            
            stripped = resp.strip()
            ends_with_punc = len(stripped) > 0 and stripped[-1] in {'.', '!', '?'}
            
            hedge_clause = extract_hedge_clause(resp)
            is_too_similar = False
            if cand["rank"] > 20:
                for prev_resp in accepted_reasonings:
                    if get_similarity(resp.lower(), prev_resp.lower()) > 0.70:
                        is_too_similar = True
                        break
                if not is_too_similar:
                    is_notice = "notice" in hedge_clause.lower()
                    thresh = 0.82 if is_notice else 0.60
                    for prev_hedge in accepted_hedges:
                        if get_similarity(hedge_clause, prev_hedge) > thresh:
                            is_too_similar = True
                            break
                        
            is_num_grounded = verify_number_grounding(resp, cand["allowed_numbers"])
            is_notice_ok = not has_notice_hallucination(resp, cand["days"])
            
            if is_grounded and is_num_grounded and is_notice_ok and not is_first_pers and not has_non_eng and has_hedge and ends_with_punc and not is_too_similar:
                reasonings[idx] = resp
                accepted_reasonings.append(resp)
                accepted_hedges.append(hedge_clause)
            else:
                still_pending.append(cand)
        pending_candidates = still_pending

    # --- ROUND 3: Fallback (Looser LLM call) ---
    if pending_candidates:
        print(f"\n--- Running fallback reasoning for {len(pending_candidates)} candidates in batches of 50 ---")
        system_fallback_rank20 = (
            "You are a professional recruiting assistant. Write exactly ONE complete sentence (under 18 words) "
            "summarising this candidate's fit for a Senior AI/ML Engineer role. "
            "Write strictly in the third person recruiter voice. Do NOT use first-person pronouns (I, my, me, myself). "
            "GOOD example: '7 years at Google building large-scale retrieval, directly matching the role's need for production search systems.' "
            "BAD example: '7 years of experience at Google, skilled in FAISS.' "
            "Reference the strongest match to the role in your sentence to show how it connects to the JD requirement. "
            "Do NOT copy skill tool names verbatim. Do NOT mention any company not listed below."
        )
        system_fallback_rank_other = (
            "You are a professional recruiting assistant. Write exactly ONE complete sentence (under 22 words) "
            "summarising this candidate's fit for a Senior AI/ML Engineer role. "
            "Write strictly in the third person recruiter voice. Do NOT use first-person pronouns (I, my, me, myself). "
            "You MUST use a contrast word like 'but' or 'however' to weave in the specific concern below. "
            "GOOD example: '7 years at Google building large-scale retrieval, but lacks recent hands-on coding experience required by the role.' "
            "BAD example: '7 years of experience at Google, skilled in FAISS.' "
            "Connect the concern to the JD requirement gap in your sentence. "
            "Note: A shorter notice period is preferred. If notice is longer than 30 days, frame it as a concern because it is longer than preferred. "
            "Do NOT copy skill tool names verbatim. Do NOT mention any company not listed below."
        )
        
        fallback_texts = []
        for cand in pending_candidates:
            skill_categories = list(dict.fromkeys(_skill_to_category(s) for s in cand["matched_skills"][:4]))
            skills_summary = ", ".join(skill_categories[:2]) if skill_categories else "applied ML systems"
            
            if cand["rank"] <= 20:
                facts = (
                    f"- Current role: {cand['title']} at {cand['company']}\n"
                    f"- Skill areas: {skills_summary}\n"
                    f"- Past roles: {cand['relevant_career']}\n"
                    f"- Strongest match to role: {cand['jd_label']}"
                )
                messages = [
                    {"role": "system", "content": system_fallback_rank20},
                    {"role": "user", "content": f"Facts:\n{facts}\n\nOutput one complete sentence ending with a period."}
                ]
            else:
                facts = (
                    f"- Current role: {cand['title']} at {cand['company']}\n"
                    f"- Skill areas: {skills_summary}\n"
                    f"- Past roles: {cand['relevant_career']}\n"
                    f"- Concern: {cand['gap']}\n"
                    f"- JD requirement gap: {cand['jd_label']}"
                )
                messages = [
                    {"role": "system", "content": system_fallback_rank_other},
                    {"role": "user", "content": f"Facts:\n{facts}\n\nRules:\n1. Output exactly one complete sentence ending with a period.\n2. You MUST use 'but' or 'however' to connect the concern to the JD requirement gap."}
                ]
            fallback_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            
        fallback_responses = []
        for i in range(0, len(fallback_texts), batch_size):
            batch_texts = fallback_texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, return_tensors="pt").to("cpu")
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            for j in range(len(batch_texts)):
                input_len = len(inputs.input_ids[j])
                resp_ids = generated_ids[j][input_len:]
                resp = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
                # Clean non-ASCII characters to keep the dynamic English output clean
                resp = re.sub(r'[^\x00-\x7f]+', '', resp).strip()
                if not resp.endswith((".", "!", "?")):
                    matches = list(re.finditer(r'[.!?](?:\s|$)', resp))
                    if matches:
                        last_punc_idx = matches[-1].start()
                        resp = resp[:last_punc_idx + 1]
                    else:
                        resp = resp + "."
                fallback_responses.append(resp)
            print(f"  Processed Fallback: {i + len(batch_texts)}/{len(pending_candidates)}")
            
        for cand, resp in zip(pending_candidates, fallback_responses):
            idx = cand["idx"]
            
            counter = 1
            while resp in accepted_reasonings or any(get_similarity(resp.lower(), r.lower()) > 0.85 for r in accepted_reasonings):
                if counter == 1 and cand.get("company"):
                    if resp.endswith("."):
                        resp = resp[:-1] + f" during tenure at {cand['company']}."
                    else:
                        resp = resp + f" during tenure at {cand['company']}."
                elif counter == 2 and cand.get("title"):
                    if resp.endswith("."):
                        resp = resp[:-1] + f" as {cand['title']}."
                    else:
                        resp = resp + f" as {cand['title']}."
                else:
                    if resp.endswith("."):
                        resp = resp[:-1] + f" (Ref: {cand['candidate_id']})."
                    else:
                        resp = resp + f" (Ref: {cand['candidate_id']})."
                counter += 1
                if counter > 5:
                    break
                    
            reasonings[idx] = resp
            accepted_reasonings.append(resp)
            accepted_hedges.append(extract_hedge_clause(resp))
                
    print(f"Time taken for LLM generation: {time.time() - llm_start_time:.2f}s")
    return reasonings
