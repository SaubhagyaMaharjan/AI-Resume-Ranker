from pydantic import BaseModel, Field
from typing import List, Optional

# This class is now in its own dedicated file for data modeling.
class JobDescription(BaseModel):
    job_title: str = Field(..., example="Junior AI Engineer")
    location: str = Field(..., example="San Francisco, CA or Remote")
    employment_type: str = Field(..., example="Full-time")
    about_the_role: str
    key_responsibilities: List[str] = Field(
        ..., min_length=1, description="A list of key responsibilities.")
    qualifications: List[str] = Field(..., min_length=1,
                                      description="A list of required qualifications.")
    certifications: Optional[List[str]] = Field(
        None, description="A list of beneficial or required certifications."
    )
    skills: Optional[dict] = Field(
        None,
        example={
            "required_skills": [
                "Proficiency in Python programming",
                "Knowledge of data preprocessing and feature engineering"
            ],
            "preferred_skills": [
                "Experience with Natural Language Processing (NLP)",
                "Knowledge of computer vision libraries (OpenCV, YOLO)"
            ]
        },
        description="Skills split into required_skills and preferred_skills lists."
    )
    what_we_offer: Optional[str] = None
    how_to_apply: Optional[str] = None

    class Config:
        # Renamed for Pydantic v2 compliance
        json_schema_extra = {
            "example": {
                "job_title": "Junior AI Engineer",
                "location": "San Francisco, CA or Remote",
                "employment_type": "Full-time",
                "about_the_role": "We are seeking a motivated and detail-oriented Junior AI Engineer...",
                "key_responsibilities": [
                    "Assist in the design, training, and evaluation of machine learning models.",
                    "Support data collection, preprocessing, and feature engineering tasks."
                ],
                "qualifications": [
                    "Bachelor’s degree in Computer Science, AI, or related field.",
                    "Proficiency in Python and familiarity with libraries such as NumPy, Pandas."
                ],
                "certifications": [
                    "Google TensorFlow Developer Certificate",
                    "AWS Certified Machine Learning – Specialty"
                ],
                "skills": {
                    "required_skills": [
                        "Proficiency in Python programming",
                        "Knowledge of data preprocessing and feature engineering"
                    ],
                    "preferred_skills": [
                        "Experience with NLP frameworks",
                        "Knowledge of computer vision libraries like OpenCV or YOLO"
                    ]
                },
                "what_we_offer": "A collaborative and supportive work environment.",
                "how_to_apply": "Please send your resume to careers@example.com."
            }
        }