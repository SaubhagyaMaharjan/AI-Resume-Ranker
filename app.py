#  1. Standard Library Imports
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List
from dotenv import load_dotenv

#  2. Third-Party Library Imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import ValidationError
import google.generativeai as genai

#  3. Local Imports
from config.logging_config import setup_logging
from job_desc.job_schema import JobDescription
from gemini_local.gemini_processor import process_cv_with_gemini
from core.resume_rank_with_gemini import (
    compute_gemini_ranking,
    json_default,
)


#  Setup Logging
setup_logging()
logger = logging.getLogger(__name__)

#  Load Gemini API Key
load_dotenv()
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise Exception(
        "GEMINI_API_KEY environment variable not set. Please create a .env file with your key."
    )

#  FastAPI App
app = FastAPI(title="Bulk Resume Validation and Ranking API")


#  API Endpoint
@app.post("/process_and_rank/")
async def process_and_rank(
    job_description_json: str = Form(
        ..., description="Structured Job Description JSON"
    ),
    files: List[UploadFile] = File(...),
):
    try:
        # Parse Job Description
        try:
            job_description_object = JobDescription.parse_raw(job_description_json)
            jd_dict = job_description_object.dict()
        except (ValidationError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid Job Description: {e}")

        # Setup Temporary Directories
        request_id = str(uuid.uuid4())
        request_temp_dir = Path("temp_uploads") / request_id
        cv_dir = request_temp_dir / "resumes"
        request_temp_dir.mkdir(parents=True, exist_ok=True)
        cv_dir.mkdir(exist_ok=True)

        #  Save Job Description as JSON
        jd_json_path = request_temp_dir / "job_description.json"
        with open(jd_json_path, "w", encoding="utf-8") as f:
            json.dump(jd_dict, f, ensure_ascii=False, indent=2, default=json_default)

        staged_cv_jsons = []
        corrupted_files = []

        # Process Resumes
        for uploaded_file in files:
            final_path = cv_dir / uploaded_file.filename
            with final_path.open("wb") as f:
                shutil.copyfileobj(uploaded_file.file, f)

            cv_json = process_cv_with_gemini(final_path, doc_type="resume")
            if cv_json and isinstance(cv_json, dict):
                cv_json["original_filename"] = uploaded_file.filename
                staged_cv_jsons.append(cv_json)

                # Save CV/Resume as JSON
                cv_json_path = cv_dir / f"{uploaded_file.filename}.json"
                with open(cv_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        cv_json, f, ensure_ascii=False, indent=2, default=json_default
                    )
            else:
                corrupted_files.append(uploaded_file.filename)

        if not staged_cv_jsons:
            raise HTTPException(
                status_code=500, detail="No valid CVs processed by Gemini."
            )

        #  Rank CVs
        rankings = compute_gemini_ranking(
            jd_json=jd_dict, cv_jsons=staged_cv_jsons, fuzzy_threshold=50
        )

        return {
            "status": "success",
            "message": "JD and CVs processed, saved as JSON, and ranked successfully.",
            "job_description_json": str(jd_json_path),
            "successfully_processed_cvs": [
                f"{cv.get('original_filename')}.json" for cv in staged_cv_jsons
            ],
            "failed_to_process": corrupted_files,
            "rankings": rankings,
        }

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected server error.")

    finally:
        logger.info(f"Temporary files saved in: {request_temp_dir}")


#  Root Endpoint
@app.get("/")
def read_root():
    return {"message": "API is running. Use /docs for details."}
