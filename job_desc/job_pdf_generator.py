import logging
from fpdf import FPDF
from pathlib import Path
from typing import List

# Import our custom Pydantic model from its new location
from job_desc.job_schema import JobDescription

logger = logging.getLogger(__name__)

# This function now lives in its own dedicated file for utility tasks.
def create_jd_pdf(jd: JobDescription, output_path: Path):
    """Creates a well-formatted PDF from a JobDescription object."""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        def write_text(text):
            return text.encode('latin-1', 'replace').decode('latin-1')

        pdf.set_font("Arial", 'B', 16)
        pdf.multi_cell(0, 10, write_text(jd.job_title), align='C')
        pdf.ln(2)

        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 8, write_text(f"{jd.location} | {jd.employment_type}"), align='C')
        pdf.ln(8)

        def render_section(title: str, content: (str | List[str])):
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, write_text(title))
            pdf.set_font("Arial", '', 11)
            if isinstance(content, str):
                pdf.multi_cell(0, 6, write_text(content))
            elif isinstance(content, list):
                for item in content:
                    pdf.multi_cell(0, 6, write_text(f"  -  {item}"))
            pdf.ln(5)

        render_section("About the Role", jd.about_the_role)
        render_section("Key Responsibilities", jd.key_responsibilities)
        render_section("Qualifications", jd.qualifications)

        if jd.certifications:
            render_section("Certifications", jd.certifications)

        if jd.skills:
            if jd.skills.get("required_skills"):
                render_section("Required Skills", jd.skills["required_skills"])
            if jd.skills.get("preferred_skills"):
                render_section("Preferred Skills", jd.skills["preferred_skills"])
        
        if jd.what_we_offer:
            render_section("What We Offer", jd.what_we_offer)
        if jd.how_to_apply:
            render_section("How to Apply", jd.how_to_apply)

        pdf.output(str(output_path))
        logger.info(f"Successfully created structured Job Description PDF at {output_path}")

    except Exception as e:
        logger.error(f"Failed to create structured Job Description PDF: {e}", exc_info=True)
        raise