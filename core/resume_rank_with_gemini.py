import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from datetime import date, datetime
import logging
import google.generativeai as genai
from typing import List, Dict, Any
from gemini_local.gemini_processor import flatten_values

logger = logging.getLogger(__name__)


# JSON serializer for date/datetime
def json_default(obj: Any) -> str:
    """
    JSON serializer for date and datetime objects.

    Args:
        obj: Any Python object.

    Returns:
        ISO-formatted string if obj is a date or datetime.

    Raises:
        TypeError: If obj is not serializable.
    """
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# Get embeddings from Gemini with fallback
def get_text_embedding(
    text: str, model: str = "gemini-text-embedding-001"
) -> np.ndarray:
    """
    Retrieve text embeddings from Gemini API with a fallback mechanism.

    Args:
        text: Input text to embed.
        model: Gemini embedding model to use (default: "gemini-text-embedding-001").

    Returns:
        Numpy array representing the text embedding vector (length 1536).

    Notes:
        - Returns a random vector if Gemini embeddings are unavailable.
        - Returns a zero vector if an exception occurs.
    """
    try:
        return np.array(
            genai.embeddings.create(model=model, input=text).data[0].embedding
        )
    except AttributeError:
        logger.warning("Gemini embeddings not available, using fallback vector.")
        return np.random.rand(1536)
    except Exception as e:
        logger.error(f"Failed to get embedding from Gemini: {e}", exc_info=True)
        return np.zeros((1536,))


# Compute ranking
def compute_gemini_ranking(
    jd_json: Dict[str, Any], cv_jsons: List[Dict[str, Any]], fuzzy_threshold: int = 70
) -> List[Dict[str, Any]]:
    """
    Compute similarity ranking between a job description (JD) and multiple resumes (CVs).

    The ranking considers:
        - Skill overlap (fuzzy matching)
        - Semantic similarity of entire text
        - Semantic similarity of skills only
        - Precision, recall, and F1 score of skills

    Args:
        jd_json: JSON dictionary representing the job description.
        cv_jsons: List of JSON dictionaries representing resumes.
        fuzzy_threshold: Threshold for fuzzy matching of skills (default 70).

    Returns:
        A sorted list of dictionaries containing scores and final ranking for each CV:
            - "resume": Original filename of the CV
            - "skill_match": Fraction of matched skills
            - "semantic_score": Cosine similarity of full text embeddings
            - "skill_semantic_score": Cosine similarity of skills embeddings
            - "precision": Skill precision score
            - "recall": Skill recall score
            - "f1_score": F1 score of skills
            - "final_score": Weighted final score
    """
    results = []

    jd_skills = flatten_values(jd_json.get("skills", {}).get("required_skills", []))
    jd_text = " ".join(flatten_values(jd_json))
    jd_embedding = get_text_embedding(jd_text).reshape(1, -1)
    jd_skills_text = " ".join(jd_skills)
    jd_skills_embedding = get_text_embedding(jd_skills_text).reshape(1, -1)

    for cv in cv_jsons:
        if cv is None:
            continue

        cv_skills = flatten_values(cv.get("skills", {}).get("technical", []))
        if not cv_skills:
            cv_skills = flatten_values(cv.get("skills", {}))

        matched_skills = set()
        for jd_skill in jd_skills:
            for cv_skill in cv_skills:
                if (
                    fuzz.partial_ratio(jd_skill.lower(), cv_skill.lower())
                    >= fuzzy_threshold
                ):
                    matched_skills.add(jd_skill)
                    break

        skill_match = len(matched_skills) / max(1, len(jd_skills))

        cv_text = " ".join(flatten_values(cv))
        cv_embedding = get_text_embedding(cv_text).reshape(1, -1)
        semantic_score = float(cosine_similarity(jd_embedding, cv_embedding)[0][0])

        cv_skills_text = " ".join(cv_skills)
        cv_skills_embedding = get_text_embedding(cv_skills_text).reshape(1, -1)
        skill_semantic_score = float(
            cosine_similarity(jd_skills_embedding, cv_skills_embedding)[0][0]
        )

        final_score = (
            0.4 * skill_match + 0.3 * semantic_score + 0.3 * skill_semantic_score
        )

        true_pos = len(matched_skills)
        false_pos = max(len(cv_skills) - true_pos, 0)
        false_neg = max(len(jd_skills) - true_pos, 0)
        precision = (
            true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        )
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results.append(
            {
                "resume": cv.get("original_filename", "unknown"),
                "skill_match": round(skill_match, 3),
                "semantic_score": round(semantic_score, 3),
                "skill_semantic_score": round(skill_semantic_score, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "final_score": round(final_score, 3),
            }
        )

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
