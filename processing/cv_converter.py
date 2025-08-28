import logging
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from paddleocr import PaddleOCR
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CONVERSION_DPI: int = 300
OCR_CONFIDENCE_THRESHOLD: float = 0.60
TEXT_THRESHOLD: int = 100

# --- OCR ENGINE INITIALIZATION ---
logger.info("Initializing PaddleOCR engine with PP-OCRv4...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4')
logger.info("PaddleOCR model loaded successfully.")


def _create_layout_preserved_text_pdf(images: List[PILImage], output_path: Path) -> bool:
    """
    Core OCR function: Creates a text-only PDF that preserves the original visual layout.
    Includes detailed debug logging for text + confidence values.
    """
    try:
        layout_pdf = fitz.open()
        total_lines_inserted = 0

        for i, image in enumerate(images):
            page = layout_pdf.new_page(width=image.width, height=image.height)
            image_np = np.array(image)

            # Run OCR
            result = ocr_engine.ocr(image_np)

            if not result or not result[0]:
                logger.warning(f"[DEBUG] No OCR result returned for page {i+1} in {output_path.name}")
                continue

            lines = result[0]
            for line_num, line in enumerate(lines, start=1):
                # Defensive check
                if not (isinstance(line, list) and len(line) == 2):
                    logger.debug(f"[DEBUG] Skipping malformed line #{line_num}: {line}")
                    continue

                box, (text, confidence) = line
                logger.info(f"[DEBUG] Page {i+1}, Line {line_num}: '{text}' (conf={confidence:.2f})")

                if confidence > OCR_CONFIDENCE_THRESHOLD and text.strip():
                    # Convert bounding box
                    x_coords = [pt[0] for pt in box]
                    y_coords = [pt[1] for pt in box]
                    rect = fitz.Rect(min(x_coords), min(y_coords),
                                     max(x_coords), max(y_coords))

                    font_size = max(6, int(rect.height * 0.8))
                    page.insert_textbox(
                        rect, text, fontsize=font_size,
                        fontname="helv", align=fitz.TEXT_ALIGN_LEFT
                    )
                    total_lines_inserted += 1
                else:
                    logger.warning(
                        f"[DEBUG] Discarded text '{text}' with confidence {confidence:.2f} "
                        f"(threshold={OCR_CONFIDENCE_THRESHOLD})"
                    )

        if total_lines_inserted == 0:
            logger.warning(f"No text with sufficient confidence was added for {output_path.name}.")
            page = layout_pdf.new_page() if len(layout_pdf) == 0 else layout_pdf[-1]
            page.insert_textbox(
                page.rect,
                "[OCR WARNING: No readable text with sufficient confidence was found.]"
            )

        layout_pdf.save(output_path, garbage=4, deflate=True, clean=True)
        layout_pdf.close()
        return True

    except Exception as e:
        logger.error(
            f"Failed during layout-preserved PDF creation for '{output_path.name}'. Reason: {e}",
            exc_info=True
        )
        return False


def convert_cvs_to_searchable_pdfs(cv_paths: List[Path]) -> List[Path]:
    """
    Processes a list of CV paths, converting any image-based or non-searchable files
    into searchable PDFs.
    """
    if not cv_paths:
        return []

    processed_paths = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

    for original_path in cv_paths:
        try:
            if original_path.suffix.lower() in image_extensions:
                logger.info(f"⚙️ Processing IMAGE file: '{original_path.name}'.")
                output_path = original_path.with_suffix(".pdf")
                image = Image.open(original_path)

                if _create_layout_preserved_text_pdf([image], output_path):
                    processed_paths.append(output_path)
                    original_path.unlink()  # remove original image
                else:
                    logger.error(f"❌ Failed to convert image: {original_path.name}")
                continue

            if original_path.suffix.lower() == '.pdf':
                doc = fitz.open(original_path)
                text_char_count = sum(len(page.get_text("text")) for page in doc)

                if text_char_count > TEXT_THRESHOLD:
                    logger.info(f"✅ Skipping already searchable PDF: '{original_path.name}'.")
                    processed_paths.append(original_path)
                    doc.close()
                    continue

                logger.info(f"⚙️ Processing non-searchable PDF: '{original_path.name}'.")
                images_to_process = []
                for page in doc:
                    pix = page.get_pixmap(dpi=CONVERSION_DPI)
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images_to_process.append(pil_image)
                doc.close()

                if _create_layout_preserved_text_pdf(images_to_process, original_path):
                    processed_paths.append(original_path)
                else:
                    logger.error(f"❌ Failed to convert non-searchable PDF: {original_path.name}")
                continue

            logger.warning(f"⚠️ Unsupported file type, skipping: '{original_path.name}'.")

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR processing '{original_path.name}'. Reason: {e}", exc_info=True)

    logger.info("--- CV Conversion Process Complete ---")
    return processed_paths