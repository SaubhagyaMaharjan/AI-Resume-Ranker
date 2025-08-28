import logging
import json
from fpdf import FPDF
from typing import Dict, Any

logger = logging.getLogger(__name__)

def create_beautiful_pdf_from_json(cv_json: Dict[str, Any], output_path: str) -> bool:
    """
    DEBUGGING FUNCTION: Takes a CV JSON object and writes its raw key-value
    structure into a simple text PDF for verification.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("Courier", size=10)

        
        json_as_string = json.dumps(cv_json, indent=4)

        # Handle potential encoding issues for FPDF's core fonts
        encoded_text = json_as_string.encode('latin-1', 'replace').decode('latin-1')
        
        # Write the entire formatted string to the PDF
        pdf.multi_cell(0, 5, txt=encoded_text)
        pdf.output(output_path)
        
        logger.info(f"Successfully rendered raw JSON data to PDF: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to render JSON to PDF. Reason: {e}", exc_info=True)
        return False