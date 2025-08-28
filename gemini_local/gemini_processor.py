import os
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from datetime import date


import fitz
import docx
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, ValidationError, Field, field_validator

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load API Key
load_dotenv()

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise Exception(
        "GEMINI_API_KEY environment variable not set. Please create a .env file with your key."
    )


# Date normalization
def normalize_date(value: Any) -> str | None:
    """
    Safely parse free-text date strings into ISO format (YYYY-MM-DD).

    Args:
        value (str): Date string, e.g., "April 2025", "Mar, 2023 - Dec, 2024"

    Returns:
        str | None: ISO date string or None if parsing fails.
    """
    if not value:
        return None

    # Already a date object
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")

    # Otherwise treat as string
    value = str(value).strip()

    for fmt in ("%Y-%m-%d", "%Y-%m", "%b, %Y", "%B, %Y", "%Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ENTITY SCHEMAS using Pydantic


class PersonalDetails(BaseModel):
    """Contact/identity fields extracted from a resume."""

    name: str
    email: Optional[str] = None
    phone: str
    linkedin: Optional[str] = None

    @field_validator("phone", mode="before")
    def normalize_phone(cls, v):
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("Phone number is required")  # enforce mandatory
        digits = re.sub(r"\D", "", v)  # keep only digits
        if digits.startswith("977"):
            return f"+{digits}"  # Already has 977
        elif digits.startswith("0"):
            return f"+977{digits[1:]}"  # Convert 0xxxxxxx to +977xxxxxxx
        elif len(digits) == 10:  # only local number
            return f"+977{digits}"
        else:
            raise ValueError(f"Invalid phone number format: {v}")


class Skills(BaseModel):
    """Structured skills section."""

    technical: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)

    @field_validator("*", mode="before")
    def normalize_skills(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v


class Education(BaseModel):
    """One education entry."""

    institution: str
    degree: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    gpa: Optional[str] = None

    @field_validator("start_date", "end_date", mode="before")
    def normalize_dates(cls, v):
        return normalize_date(v)


class Experience(BaseModel):
    """One work experience entry."""

    role: str
    company: str
    location: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    description: List[str] = Field(default_factory=list)

    @field_validator("start_date", "end_date", mode="before")
    def normalize_dates(cls, v):
        return normalize_date(v)


class Certification(BaseModel):
    """Represents a single certification in a resume."""

    name: str
    issuer: Optional[str] = None
    issue_date: Optional[date] = None
    credential_id: Optional[str] = None

    @field_validator("issue_date", mode="before")
    def normalize_dates(cls, v):
        return normalize_date(v)


class Project(BaseModel):
    """One project entry."""

    name: str
    description: Optional[str] = None
    tech_stack: List[str] = Field(default_factory=list)


class ResumeSchema(BaseModel):
    """Structured resume payload."""

    personal_details: PersonalDetails
    summary: Optional[str] = None
    skills: Skills = Skills()
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    other_sections: Dict[str, List[str]] = Field(default_factory=dict)


class JobDescriptionSchema(BaseModel):
    """Structured job description payload."""

    job_title: str
    company: str
    location: Optional[str] = None
    employment_type: Optional[str] = None  # e.g., Full-time, Part-time, Contract
    salary: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    skills_required: List[str] = Field(default_factory=list)
    experience_level: Optional[str] = None  # e.g., Junior, Mid, Senior
    benefits: List[str] = Field(default_factory=list)
    other_sections: Dict[str, List[str]] = Field(default_factory=dict)

    @field_validator("skills_required")
    @classmethod
    def validate_skills_required(cls, value):
        if not value or not any(skill.strip() for skill in value):
            raise ValueError("At least one non-empty skill is required")
        return [skill.strip() for skill in value]

    @field_validator("responsibilities", "qualifications", "benefits")
    @classmethod
    def validate_lists(cls, value):
        if value and not any(item.strip() for item in value):
            raise ValueError("List items cannot be empty")
        return value


# --- MODEL INITIALIZATION ---
MODEL_NAME = "gemini-1.5-flash-latest"
model = genai.GenerativeModel(MODEL_NAME)

# --- THE INTELLIGENT PROMPT FOR STRUCTURED JSON ---
RESUME_PROMPT = """
You are an expert HR data extraction assistant. Analyze the provided resume image(s) and convert the content into a structured, valid JSON object following this exact schema:
{
  "personal_details": {"name": "Full Name", "email": "email@address.com", "phone": "Phone Number", "linkedin": "LinkedIn URL/username"},
  "summary": "The professional summary or objective text.",
  "experience": [{"role": "Job Title", "company": "Company Name", "location": "City, State", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "description": ["Responsibility 1.", "Accomplishment 2."]}],
  "education": [{"institution": "University Name", "degree": "Degree and Major", "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD", "gpa": "GPA"}],
  "skills": {"technical": ["Skill 1", "Skill 2"],"tools": ["Tool 1", "Tool 2"],"languages": ["Language 1", "Language 2"],"soft": ["Soft skill 1", "Soft skill 2"]},
  "certifications": [{"name": "Certification Name","issuer": "Issuer Name", "issue_date": "YYYY-MM-DD","credential_id": "Optional ID"}],
  "other_sections": {"Section Title 1": ["Bullet point 1"], "Section Title 2": ["Bullet point 2"]}
}
**Rules:**
- Analyze the layout carefully to correctly associate text with its proper section.
- For sections like 'VOLUNTEER' or 'CERTIFICATIONS', place them in the `other_sections` object.
- If a section or field is not found, use an empty list [], empty object {}, or empty string "".
- The final output MUST be a single, valid JSON object inside a ```json ... ``` block. Do not include any other text.
"""

JD_PROMPT = """
You are an expert HR data extraction assistant. Analyze the provided job description (image or text)
and convert it into a structured, valid JSON object following this schema:
{
  "job_title": "Software Engineer",
  "company": "Company Name",
  "location": "City, State or Remote",
  "employment_type": "Full-time/Part-time/Contract",
  "salary": "Salary range if available",
  "description": "Overall job description",
  "responsibilities": ["Responsibility 1", "Responsibility 2"],
  "qualifications": ["Qualification 1", "Qualification 2"],
  "skills_required": ["Skill 1", "Skill 2"],
  "benefits": ["Benefit 1", "Benefit 2"],
  "other_sections": {"Section Title": ["Item 1"]}
}
Rules:
- If a section is not present, leave it empty (\"\", [], or {}).
- The final output MUST be a single, valid JSON object inside a ```json ... ``` block. Do not include any other text.
"""


# File Processing


def _get_images_from_file(file_path: Path) -> tuple[str, List[Image.Image]]:
    """
    Reads text content or converts PDF/image files to PIL Images.

    Args:
        file_path: Path to the PDF/PNG/JPG/JPEG file.

    Returns:
        tuple: (text_content, images)
            - text_content: str for .txt/.docx files, empty string for PDFs/images
            - images: List of PIL Images for PDFs/images, empty list for text files
    """
    text_content = ""
    images: List[Image.Image] = []
    ext = file_path.suffix.lower()
    image_extensions = [".png", ".jpg", "jpeg"]
    if ext in [".txt", ".docx"]:
        try:
            if ext == ".txt":
                text_content = file_path.read_text(encoding="utf-8")
            else:  # .docx
                doc = docx.Document(file_path)
                text_content = "\n".join(
                    [p.text for p in doc.paragraphs if p.text.strip()]
                )
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")

    elif ext in [".pdf", ".png", ".jpg", ".jpeg"]:
        image_extensions = [".png", ".jpg", "jpeg"]
        if ext in image_extensions:
            try:
                images.append(Image.open(file_path))
            except Exception as e:
                logger.error(f"Failed to read image '{file_path.name}': {e}")
        elif ext == ".pdf":
            try:
                doc = fitz.open(file_path)
                for page in doc:
                    pix = page.get_pixmap(dpi=200)
                    images.append(
                        Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    )
                doc.close()
            except Exception as e:
                logger.error(f"Failed to read PDF '{file_path.name}': {e}")
    else:
        logger.warning(f"Unsupported file type: {file_path.suffix}")

    return text_content, images


def extract_json_from_response(
    response_text: str, file_name: str
) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from a Gemini text response using robust fenced-block parsing.
    Handles:
      - Fenced code blocks (```json ... ```)
      - Extra text before/after JSON
      - Minor formatting errors like single quotes, trailing commas

    Args:
        response_text: Raw text from Gemini response.
        file_name: For logging context.

    Returns:
        Parsed JSON dict if found and valid, otherwise None.
    """
    try:
        json_block_match = re.search(
            r"```json(.*?)```", response_text, re.DOTALL | re.IGNORECASE
        )
        if json_block_match:
            json_string = json_block_match.group(1).strip()
        else:
            # Fallback: remove any surrounding triple backticks
            json_string = response_text.strip("` \n")

        # Replace single quotes with double quotes for JSON compatibility
        json_string = json_string.replace("'", '"')

        # Remove trailing commas before closing brackets/braces
        json_string = re.sub(r",\s*(\]|})", r"\1", json_string)

        parsed_json = json.loads(json_string)
        logger.info(f"Successfully parsed structured JSON from '{file_name}'.")
        return parsed_json

    except Exception as e:
        logger.error(
            f"Failed to extract/parse JSON from '{file_name}'. Raw response:\n{response_text}\nError: {e}",
            exc_info=True,
        )
        return None


def validate_payload(
    payload: Dict[str, Any], doc_type: Literal["resume", "job_description"]
) -> Optional[Dict[str, Any]]:
    """
    Validate the parsed JSON payload against the appropriate Pydantic schema.

    Args:
        payload: Parsed dict from the Gemini response.
        doc_type: Either "resume" or "job_description".

    Returns:
        A normalized dict (via Pydantic .model_dump()) if valid, else None.
    """
    try:
        if doc_type == "resume":
            return ResumeSchema(**payload).model_dump(mode="json")
        if doc_type == "job_description":
            return JobDescriptionSchema(**payload).model_dump(mode="json")
        raise ValueError("doc_type must be 'resume' or 'job_description'")
    except ValidationError as ve:
        logger.error(f"Schema validation failed for {doc_type}: {ve.errors()}")
        return None


def process_cv_with_gemini(
    file_path: Path,
    doc_type: Literal["resume", "job_description"],
) -> Optional[Dict[str, Any]]:
    """
    Processes a CV (image or PDF) using the Gemini API to extract its content
    into a structured JSON object. Returns the parsed dictionary. On failure,
    returns a minimal JSON object for resumes or None for job descriptions.

    Args:
        file_path: Path to input PDF/PNG/JPG/JPEG file.
        doc_type (str): "resume" or "job_description".

    Returns:
       Dict[str, Any]: The validated JSON dict, or minimal dict if resume parsing fails.
    """

    logger.info(f"Parsing '{file_path.name}' with Gemini API for structured JSON...")

    text_content, images = _get_images_from_file(file_path)

    if not text_content and not images:
        logger.warning(f"Could not extract content from: {file_path.name}")
        if doc_type == "resume":
            return {
                "original_filename": file_path.name,
                "parse_success": False,
                "skills": [],
                "experience": [],
                "education": [],
                "contact": {},
                "summary": "",
                "additional_info": {},
            }
        return None

    prompt = RESUME_PROMPT if doc_type == "resume" else JD_PROMPT

    try:
        response = model.generate_content(
            [prompt] + images
            if images
            else [prompt] + ([text_content] if text_content else []),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    except Exception as e:
        logger.error(f"Gemini API call failed for {file_path.name}: {e}", exc_info=True)
        if doc_type == "resume":
            return {
                "original_filename": file_path.name,
                "parse_success": False,
                "skills": [],
                "experience": [],
                "education": [],
                "contact": {},
                "summary": "",
                "additional_info": {},
            }
        return None

    response_text = response.text.strip()
    parsed = extract_json_from_response(response_text, file_path.name)
    if not parsed:
        logger.error(f"Gemini failed to produce valid JSON for {file_path.name}")
        if doc_type == "resume":
            return {
                "original_filename": file_path.name,
                "parse_success": False,
                "skills": [],
                "experience": [],
                "education": [],
                "contact": {},
                "summary": "",
                "additional_info": {},
            }
        return None

    # Normalize parsed JSON
    try:
        if doc_type == "resume":
            parsed = normalize_resume_data(parsed)
            parsed = validate_resume_data(parsed)
        elif doc_type == "job_description":
            parsed = normalize_job_description_data(parsed)
            parsed = validate_job_data(parsed)
    except ValueError as ve:
        logger.error(f"Normalization error for {file_path.name}: {ve}")
        if doc_type == "resume":
            return {
                "original_filename": file_path.name,
                "parse_success": False,
                "skills": [],
                "experience": [],
                "education": [],
                "contact": {},
                "summary": "",
                "additional_info": {},
            }
        return None

    validated = validate_payload(parsed, doc_type)
    if not validated:
        logger.error(f"Schema validation failed for {file_path.name}")
        if doc_type == "resume":
            return {
                "original_filename": file_path.name,
                "parse_success": False,
                "skills": [],
                "experience": [],
                "education": [],
                "contact": {},
                "summary": "",
                "additional_info": {},
            }
        return None

    parsed["original_filename"] = file_path.name
    parsed["parse_success"] = True

    logger.info(f"Normalized and validated JSON for {file_path.name}")
    return parsed


def normalize_resume_data(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a parsed resume JSON to match the ResumeSchema expectations.

    Ensures:
    - 'personal_details' exists with mandatory 'name' and 'phone'.
    - Skills, education, experience, certifications, projects, and other_sections
      have consistent types and default values.
    - Strings are stripped of leading/trailing whitespace.
    - Experience descriptions and project/other_sections are lists of strings.
    - Dates are normalized if possible.

    Args:
        parsed (Dict[str, Any]): The raw JSON dict returned from Gemini API.

    Returns:
        Dict[str, Any]: The normalized JSON dict ready for schema validation.

    Raises:
        ValueError: If mandatory fields (like phone) are missing.
    """

    # Ensure personal_details exists
    pd = parsed.get("personal_details", {})
    if "name" not in pd or not pd["name"]:
        pd["name"] = "Unknown Name"
    if "phone" not in pd or not pd["phone"]:
        pd["phone"] = "N/A"
    pd.setdefault("email", "")
    pd.setdefault("linkedin", "")
    parsed["personal_details"] = pd

    # Normalize skills
    skills = parsed.get("skills", {})
    for key in ["technical", "tools", "languages", "soft"]:
        val = skills.get(key)
        if not isinstance(val, list):
            skills[key] = []
        else:
            skills[key] = [str(v).strip() for v in val if v]
    parsed["skills"] = skills

    # Normalize education
    education_list = parsed.get("education", [])
    for edu in education_list:
        edu.setdefault("institution", "")
        edu.setdefault("degree", "")
        edu.setdefault("gpa", "")
        dates = edu.get("dates", "")
        if "-" in dates:
            start, end = map(str.strip, dates.split("-"))
        else:
            start, end = dates.strip() or None, None
        edu["start_date"] = normalize_date(edu.get("start_date") or start)
        edu["end_date"] = normalize_date(edu.get("end_date") or end)
    parsed["education"] = education_list

    # Normalize experience
    exp_list = parsed.get("experience", [])
    for exp in exp_list:
        exp.setdefault("role", "")
        exp.setdefault("company", "")
        exp.setdefault("location", "")
        exp.setdefault("description", [])
        if isinstance(exp["description"], str):
            exp["description"] = [exp["description"]]
        elif not isinstance(exp["description"], list):
            exp["description"] = []

        dates = exp.get("dates", "")
        if "-" in dates:
            start, end = map(str.strip, dates.split("-"))
        else:
            start, end = dates.strip() or None, None
        exp["start_date"] = normalize_date(exp.get("start_date") or start)
        exp["end_date"] = normalize_date(exp.get("end_date") or end)
    parsed["experience"] = exp_list

    # Normalize certifications
    cert_list = parsed.get("certifications", [])
    for cert in cert_list:
        cert.setdefault("name", "")
        cert.setdefault("issuer", "")
        cert.setdefault("credential_id", "")
        issue_date = cert.get("issue_date", "")
        cert["issue_date"] = normalize_date(issue_date)
    parsed["certifications"] = cert_list

    # Normalize projects
    projects = parsed.get("projects", [])
    for proj in projects:
        proj.setdefault("name", "")
        proj.setdefault("description", "")
        proj.setdefault("tech_stack", [])
        if not isinstance(proj["tech_stack"], list):
            proj["tech_stack"] = []
    parsed["projects"] = projects

    # Normalize other_sections
    other_sections = parsed.get("other_sections", {})
    for section, items in other_sections.items():
        if not isinstance(items, list):
            other_sections[section] = []
        else:
            normalized_items = []
            for item in items:
                if isinstance(item, dict):
                    # Combine all dict values into a single string
                    normalized_items.append(" — ".join(str(v) for v in item.values()))
                else:
                    normalized_items.append(str(item))
            other_sections[section] = normalized_items
    parsed["other_sections"] = other_sections

    return parsed


def normalize_job_description_data(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a parsed job description JSON to match the JobDescriptionSchema expectations.

    Ensures:
    - Mandatory fields are present with default empty strings.
    - Lists like responsibilities, qualifications, skills_required, benefits are proper lists of strings.
    - 'skills_required' are lowercased and unique.
    - 'other_sections' is a dict of lists of strings.

    Args:
        parsed (Dict[str, Any]): The raw JSON dict returned from Gemini API.

    Returns:
        Dict[str, Any]: The normalized JSON dict ready for schema validation.
    """

    # Basic fields
    for field in [
        "job_title",
        "company",
        "location",
        "employment_type",
        "salary",
        "description",
        "experience_level",
    ]:
        val = parsed.get(field)
        parsed[field] = str(val).strip() if val else ""

    # List fields
    for list_field in [
        "responsibilities",
        "qualifications",
        "skills_required",
        "benefits",
    ]:
        val = parsed.get(list_field)
        if not isinstance(val, list):
            parsed[list_field] = []
        else:
            parsed[list_field] = [str(v).strip() for v in val if v]

    # Deduplicate and lowercase skills_required
    if "skills_required" in parsed:
        parsed["skills_required"] = sorted(
            set(s.lower() for s in parsed["skills_required"])
        )

    # --- Other Sections ---
    other_sections = parsed.get("other_sections", {})
    if not isinstance(other_sections, dict):
        other_sections = {}
    for section, items in other_sections.items():
        if not isinstance(items, list):
            other_sections[section] = []
        else:
            normalized_items = []
            for item in items:
                if isinstance(item, dict):
                    # Convert dict values to string joined by " — "
                    normalized_items.append(" — ".join(str(v) for v in item.values()))
                else:
                    normalized_items.append(str(item))
            other_sections[section] = normalized_items
    parsed["other_sections"] = other_sections

    return parsed


def validate_resume_data(data: dict) -> dict | None:
    """
    Validate a resume dictionary against the ResumeSchema.

    Args:
        data (dict): The parsed and normalized resume JSON dictionary.

    Returns:
        dict | None: A validated resume dictionary if validation succeeds,
        otherwise None if validation fails.
    """
    try:
        return ResumeSchema(**data).dict()
    except ValidationError as e:
        logger.error(f" Resume validation failed: {e}")
        return None


def validate_job_data(data: dict) -> dict | None:
    """
    Validate a job description dictionary against the JobDescriptionSchema.

    Args:
        data (dict): The parsed and normalized job description JSON dictionary.

    Returns:
        dict | None: A validated job description dictionary if validation succeeds,
        otherwise None if validation fails.
    """
    try:
        return JobDescriptionSchema(**data).dict()
    except ValidationError as e:
        logger.error(f"Job description validation failed: {e}")
        return None


# for evaluation metrices


def flatten_values(obj) -> set:
    out = set()
    if obj is None:
        return out
    if isinstance(obj, str):
        s = obj.strip().lower()
        if s:
            out.add(s)
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out |= flatten_values(v)
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out |= flatten_values(v)
        return out
    return out
