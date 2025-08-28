import logging
from pathlib import Path
from typing import List, Dict, Any
import unicodedata

import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
MODEL_NAME = "all-MiniLM-L6-v2"


# This function is called ONCE at startup by app.py
def load_ranking_model() -> SentenceTransformer:
    """Loads the Sentence Transformer model into memory."""
    logger.info(f"Loading Sentence Transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Sentence Transformer model loaded successfully.")
    return model


def _normalize_text(text: str) -> str:
    """Cleans and normalizes text for consistent AI model processing."""
    if not text:
        return ""
    normalized_text = unicodedata.normalize("NFKC", text)
    return " ".join(normalized_text.lower().split())


def _reconstruct_text_from_json(cv_json: Dict[str, Any]) -> str:
    """Creates a clean, coherent text block from structured JSON for semantic comparison."""
    text_parts = []

    # Prioritize key sections for semantic meaning
    if cv_json.get("summary"):
        text_parts.append(cv_json["summary"])

    if cv_json.get("work_experience"):
        text_parts.append("\nWork Experience:")
        for exp in cv_json.get("work_experience", []):
            text_parts.append(f"{exp.get('role', '')} at {exp.get('company', '')}")
            if exp.get("description"):
                text_parts.extend([f"- {d}" for d in exp["description"]])

    if cv_json.get("skills"):
        text_parts.append("\nSkills: " + ", ".join(cv_json.get("skills", [])))

    return _normalize_text("\n".join(text_parts).strip())


def _calculate_skill_score(jd_skills: List[str], cv_skills: List[str]) -> float:
    """Calculates a factual score (0.0 to 1.0) based on the overlap of skills."""
    if not jd_skills:
        return 1.0  # If JD requires no skills, it's a perfect match.

    # Use sets for efficient, case-insensitive comparison
    required_skills_set = {skill.lower().strip() for skill in jd_skills}
    candidate_skills_set = {skill.lower().strip() for skill in cv_skills}

    matching_skills = required_skills_set.intersection(candidate_skills_set)

    score = len(matching_skills) / len(required_skills_set)
    return min(score, 1.0)  # Ensure score doesn't exceed 1.0


# --- The Main Ranking Function ---
def rank_resumes(
    model: SentenceTransformer, jd_json: Dict[str, Any], cv_jsons: List[Dict]
) -> List[Dict]:
    """
    Ranks CVs (as JSON) against a JD (as JSON) using a hybrid scoring system.
    """
    if not cv_jsons:
        return []

    # --- 1. Prepare Job Description Data ---
    jd_text_for_semantic = _reconstruct_text_from_json(jd_json)
    # Use the 'required_skills' from the 'skills' object if it exists
    jd_required_skills = jd_json.get("skills", {}).get(
        "required_skills", jd_json.get("skills", [])
    )

    if not jd_text_for_semantic:
        logger.error("Could not reconstruct text from Job Description JSON.")
        return []

    # --- 2. Prepare CV Data ---
    cv_texts_for_semantic = [_reconstruct_text_from_json(cv) for cv in cv_jsons]
    original_filenames = [cv.get("original_filename", "Unknown") for cv in cv_jsons]

    # --- 3. Calculate Semantic Scores (in one batch for efficiency) ---
    logger.info("Calculating semantic similarity scores...")
    jd_vector = model.encode([jd_text_for_semantic])
    cv_vectors = model.encode(cv_texts_for_semantic)
    semantic_scores = cosine_similarity(jd_vector, cv_vectors)[0]

    # --- 4. Calculate Final Hybrid Score for Each CV ---
    ranked_results = []
    for i, cv_json in enumerate(cv_jsons):
        semantic_score = semantic_scores[i]
        skill_score = _calculate_skill_score(
            jd_required_skills, cv_json.get("skills", [])
        )

        # --- 5. Combine Scores with a Weighted Average (Tunable) ---
        semantic_weight = 0.6
        skill_weight = 0.4
        final_score = (semantic_weight * semantic_score) + (skill_weight * skill_score)

        ranked_results.append(
            {
                "cv_filename": original_filenames[i],
                "final_score": float(final_score),
                "details": {
                    "semantic_match": float(semantic_score),
                    "skill_match": float(skill_score),
                },
            }
        )

    # --- 6. Sort by the Final Hybrid Score ---
    ranked_results.sort(key=lambda x: x["final_score"], reverse=True)

    logger.info(f"Successfully ranked {len(ranked_results)} CVs.")
    return ranked_results
