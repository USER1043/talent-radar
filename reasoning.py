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
    
    # List of all known skills in the dataset
    all_known_skills = [
        "langchain", "llamaindex", "pinecone", "weaviate", "qdrant", "milvus", 
        "opensearch", "elasticsearch", "faiss", "python", "pytorch", "tensorflow", 
        "scikit-learn", "numpy", "pandas", "lora", "qlora", "peft", "xgboost", 
        "nlp", "computer vision", "opencv", "cnn", "yolo", "gans", "diffusion models"
    ]
    allowed_skills_lower = [s.lower() for s in allowed_skills]
    
    for s in all_known_skills:
        if s in text_lower and s not in allowed_skills_lower:
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
    allowed_cos_lower = [c.lower() for c in allowed_companies]
    
    for c in known_cos:
        if c in text_lower and c not in allowed_cos_lower:
            return False
            
    return True


# Fallback generation using a looser LLM prompt that forbids specific names/skills to guarantee grounding.
def generate_looser_reasoning(model, tokenizer, yoe: float, title: str, rank: int, gap: str) -> str:
    if rank <= 20:
        messages = [
            {"role": "system", "content": "You are a professional recruiting assistant. Summarize this candidate's fit in exactly one short, complete sentence (under 12 words). Do NOT mention any specific company names or technical skill names. Just describe their role, experience, and rank."},
            {"role": "user", "content": f"Facts:\n- Role: {title}\n- Experience: {yoe} years\n- Rank: #{rank} of 100\n\nRules:\n1. Limit output to 12 words.\n2. Do NOT mention any specific company names or technical tools.\n3. Output a complete sentence."}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a professional recruiting assistant. Summarize this candidate's fit in exactly one short, complete sentence (under 15 words) politely mentioning the concern. Do NOT mention any specific company names or technical skill names."},
            {"role": "user", "content": f"Facts:\n- Role: {title}\n- Experience: {yoe} years\n- Rank: #{rank} of 100\n- Concern: {gap}\n\nRules:\n1. Limit output to 15 words.\n2. Do NOT mention any specific company names or technical tools.\n3. Politely mention the concern.\n4. Output a complete sentence."}
        ]
        
    with torch.no_grad():
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=35,
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
        
        # Weakest matching dimension
        gap = "minor experience gaps relative to senior requirements"
        if row.get("no_recent_coding") == 1.0:
            gap = "no recent production coding in the last 18 months"
        elif row.get("is_consulting_only") == 1.0:
            gap = "career entirely at consulting firms"
        elif row.get("cv_speech_robotics_no_nlp") == 1.0:
            gap = "computer vision/speech background with no NLP exposure"
        elif row.get("is_pure_research") == 1.0:
            gap = "pure research background with no production deployment"
        elif row.get("notice_period_penalty", 0.0) > 0.2:
            gap = f"long notice period of {days} days"
        elif row.get("semantic_sim_to_ideal", 1.0) < 0.4:
            gap = "lower alignment with ideal candidate profile"
        elif row.get("behavioral_score", 1.0) < 0.4:
            gap = "low recruiter response rate and profile activity"
            
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
        
        # Verify grounding and fallback to a looser LLM call if ungrounded or empty
        if not response or not verify_grounding(response, allowed_skills, allowed_cos):
            response = generate_looser_reasoning(model, tokenizer, yoe, title, rank, gap)
                
        reasonings.append(response)
        
    return reasonings
