"""
@file Generates grounded reasoning statements for the top-ranked candidates using a local LLM.
@package online_ranking
"""

from __future__ import annotations

import json
import re
import torch
import pandas as pd

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


# Fallback generation using a looser LLM prompt that references actual resume facts but bans specific tool/company names.
def generate_looser_reasoning(
    model, tokenizer,
    yoe: float, title: str, company: str, rank: int, gap: str,
    matched_skills: list[str], past_roles: str
) -> str:
    # Convert specific skill names to safe category labels
    skill_categories = list(dict.fromkeys(_skill_to_category(s) for s in matched_skills[:4]))
    skills_summary = ", ".join(skill_categories[:2]) if skill_categories else "applied ML systems"

    if rank <= 20:
        facts = (
            f"- Current role: {title} at {company}\n"
            f"- Experience: {yoe} years\n"
            f"- Skill areas: {skills_summary}\n"
            f"- Past roles: {past_roles}\n"
            f"- Rank: #{rank} of 100"
        )
        messages = [
            {"role": "system", "content": (
                "You are a professional recruiting assistant. Write exactly ONE complete sentence (under 18 words) "
                "summarising this candidate's fit for a Senior AI/ML Engineer role, using the facts below. "
                "Reference their role, experience, or skill areas — do NOT copy skill tool names verbatim (e.g. write "
                "'vector search' not 'FAISS'). Do NOT mention any company not listed below."
            )},
            {"role": "user", "content": f"Facts:\n{facts}\n\nOutput one complete sentence ending with a period."}
        ]
    else:
        facts = (
            f"- Current role: {title} at {company}\n"
            f"- Experience: {yoe} years\n"
            f"- Skill areas: {skills_summary}\n"
            f"- Past roles: {past_roles}\n"
            f"- Rank: #{rank} of 100\n"
            f"- Concern: {gap}"
        )
        messages = [
            {"role": "system", "content": (
                "You are a professional recruiting assistant. Write exactly ONE complete sentence (under 22 words) "
                "summarising this candidate's fit for a Senior AI/ML Engineer role, weaving in a description of the "
                "specific concern below in your own words — do NOT copy the concern phrase verbatim. "
                "Reference their role, experience, or skill areas. Do NOT copy skill tool names verbatim. "
                "Do NOT mention any company not listed below."
            )},
            {"role": "user", "content": f"Facts:\n{facts}\n\nOutput one complete sentence ending with a period that describes the concern naturally."}
        ]
        
    with torch.no_grad():
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=50,
            temperature=0.1,
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
            
    return response


# Generates reasoning statements for the final 100 candidates.
def generate_reasonings(candidates_df: pd.DataFrame) -> list[str]:
    model, tokenizer = _get_model_and_tokenizer()
    reasonings = []
    
    # Define JD key skills
    jd_skills = {
        "sentence-transformers", "embeddings", "vector search", "hybrid search", 
        "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", 
        "faiss", "python", "ndcg", "mrr", "map", "a/b testing", "lora", "qlora", 
        "peft", "learning to rank", "xgboost", "neural ranking", "hr-tech", 
        "distributed systems", "nlp", "information retrieval"
    }

    for _, row in candidates_df.iterrows():
        yoe = row["years_of_experience"]
        title = row["current_title"]
        company = row["current_company"]
        rank = row["rank"]
        
        # Parse skills
        try:
            skills_list = [s["name"] for s in json.loads(row["skills_json"])]
        except (json.JSONDecodeError, TypeError, KeyError):
            skills_list = []
        
        matched_skills = [s for s in skills_list if s.lower() in jd_skills or any(x in s.lower() for x in ["embeddings", "vector", "search", "ranking", "eval", "llm", "finetun", "fine-tun", "nlp", "retrieval"])]
        if not matched_skills:
            matched_skills = skills_list[:5]
            
        # Parse career
        try:
            career_list = json.loads(row["career_history_json"])
        except (json.JSONDecodeError, TypeError):
            career_list = []
            
        roles = []
        companies = [company]
        for c in career_list[:2]:
            roles.append(f"{c.get('title')} at {c.get('company')} ({c.get('duration_months', 0)}mo)")
            if c.get("company"):
                companies.append(c.get("company"))
        relevant_career = "; ".join(roles)
        
        # Engagement
        recency = row["redrob_last_active_date"]
        rate = row["redrob_recruiter_response_rate"]
        days = row["redrob_notice_period_days"]
        
        # Weakest matching dimension — use actual score columns to be specific
        gap = None
        if row.get("no_recent_coding") == 1.0:
            gap = "has not been in a hands-on coding role for the past 18+ months"
        elif row.get("is_consulting_only") == 1.0:
            gap = "career has been entirely at consulting firms with no product company experience"
        elif row.get("cv_speech_robotics_no_nlp") == 1.0:
            gap = "background is in computer vision / speech with no NLP or IR exposure"
        elif row.get("is_pure_research") == 1.0:
            gap = "comes from a pure research background with no shipped production systems"
        elif row.get("notice_period_penalty", 0.0) > 0.2:
            gap = f"notice period of {int(days)} days is longer than the JD's preferred 30-day buy-out"
        
        # If no hard flag fired, derive a concrete gap from the weakest score column
        if gap is None:
            sem_ideal = float(row.get("semantic_sim_to_ideal", 1.0))
            beh = float(row.get("behavioral_score", 1.0))
            yoe_val = float(yoe)
            if yoe_val < 4:
                gap = f"only {yoe_val:.1f} years of experience, below the JD's 6-8 year target"
            elif yoe_val > 9:
                gap = f"{yoe_val:.1f} years of experience without matching seniority markers (open-source / publications)"
            elif sem_ideal < 0.45:
                gap = "profile narrative does not closely match the ideal senior AI/search engineer archetype"
            elif beh < 0.45:
                gap = "low recent platform activity and recruiter response rate on the platform"
            elif sem_ideal < 0.55:
                gap = "moderate alignment with the ideal candidate narrative, with room to grow into a senior scope"
            else:
                gap = "experience range and skill mix are solid but do not yet demonstrate the end-to-end system ownership the role requires"
            
        # Allowed lists for validation
        allowed_skills = skills_list
        allowed_cos = list(set(companies))
        
        # Prompt
        messages = [
            {"role": "system", "content": "You are a professional recruiting assistant. Summarize this candidate's fit in exactly one short, complete sentence (under 20 words) using ONLY the verified facts. Do not assume or invent anything."},
            {"role": "user", "content": f"Facts:\n- Experience: {yoe} years\n- Role: {title} at {company}\n- Skills: {', '.join(matched_skills[:3])}\n- Past: {relevant_career}\n- Rank: #{rank} of 100\n- Concern: {gap}\n\nRules:\n1. Output exactly one complete sentence ending with a period.\n2. Do not hallucinate any details.\n3. If rank is below #20, you must politely mention the concern."}
        ]
        
        # Run model under no_grad for speed and memory efficiency
        with torch.no_grad():
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
            
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=45,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Clean trailing cut-off sentence fragment (avoid truncating at decimal points)
        if not response.endswith((".", "!", "?")):
            matches = list(re.finditer(r'[.!?](?:\s|$)', response))
            if matches:
                last_punc_idx = matches[-1].start()
                response = response[:last_punc_idx + 1]
            else:
                response = ""  # Force fallback if no sentence ending exists
        
        # Verify grounding and retry once with a stricter prompt before falling back to loose LLM
        if not response or not verify_grounding(response, allowed_skills, allowed_cos):
            messages_retry = [
                {"role": "system", "content": "You are a professional recruiting assistant. Summarize this candidate's fit in exactly one short, complete sentence (under 20 words). You MUST strictly use ONLY the verified facts. Do not assume or invent anything. Do not mention any companies or skills not in the list below."},
                {"role": "user", "content": f"Facts:\n- Experience: {yoe} years\n- Role: {title} at {company}\n- Skills: {', '.join(matched_skills[:3])}\n- Past: {relevant_career}\n- Rank: #{rank} of 100\n- Concern: {gap}\n\nRules:\n1. Output exactly one complete sentence ending with a period.\n2. Do NOT hallucinate. Use ONLY the listed skills and companies.\n3. If rank is below #20, politely mention the concern."}
            ]
            with torch.no_grad():
                text_retry = tokenizer.apply_chat_template(messages_retry, tokenize=False, add_generation_prompt=True)
                model_inputs_retry = tokenizer([text_retry], return_tensors="pt").to("cpu")
                generated_ids_retry = model.generate(
                    model_inputs_retry.input_ids,
                    max_new_tokens=45,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_ids_retry = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs_retry.input_ids, generated_ids_retry)
                ]
                response = tokenizer.batch_decode(generated_ids_retry, skip_special_tokens=True)[0].strip()
            
            if not response.endswith((".", "!", "?")):
                matches = list(re.finditer(r'[.!?](?:\s|$)', response))
                if matches:
                    last_punc_idx = matches[-1].start()
                    response = response[:last_punc_idx + 1]
                else:
                    response = ""
            
            # If the retry also fails grounding verification, fall back to the looser second-tier LLM call
            if not response or not verify_grounding(response, allowed_skills, allowed_cos):
                response = generate_looser_reasoning(
                    model, tokenizer, yoe, title, company, rank, gap,
                    matched_skills, relevant_career
                )
                
        reasonings.append(response)
        
    return reasonings
