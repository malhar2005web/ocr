import os
import io
import logging
from typing import Dict, Any, List, Tuple
from PIL import Image
import fitz  # PyMuPDF

from app.core.config import settings
from app.services.ocr.engine import TesseractOCREngine

logger = logging.getLogger(__name__)

class DocumentExtractor:
    def __init__(self):
        self.ocr_engine = TesseractOCREngine(tesseract_cmd=settings.TESSERACT_CMD)

    def extract_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Determines file format and extracts text, layout, and tables.
        Uses native PDF text extraction first; seamlessly falls back to OCR for scanned pages/images.
        """
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext == "pdf":
            return self._extract_pdf(file_path, filename)
        elif ext in ["jpg", "jpeg", "png"]:
            return self._extract_image(file_path, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: {settings.ALLOWED_EXTENSIONS}")

    def _extract_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        doc = fitz.open(file_path)
        page_count = doc.page_count
        pages_content = []
        raw_text_parts = []
        all_tables = []
        used_ocr = False
        used_native = False

        for page_num in range(page_count):
            page = doc[page_num]
            # 1. Attempt native text extraction
            text = page.get_text("text").strip()
            
            # Extract structured tables if available via PyMuPDF
            page_tables = []
            try:
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for t_idx, tab in enumerate(tabs.tables):
                        extracted_tab = tab.extract()
                        if extracted_tab and len(extracted_tab) > 1:
                            headers = [str(c or "").strip() for c in extracted_tab[0]]
                            rows = []
                            for row in extracted_tab[1:]:
                                row_dict = {}
                                for idx, val in enumerate(row):
                                    header_name = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx+1}"
                                    row_dict[header_name] = str(val or "").strip()
                                rows.append(row_dict)
                            page_tables.append({
                                "table_id": f"p{page_num + 1}_t{t_idx + 1}",
                                "page": page_num + 1,
                                "headers": headers,
                                "rows": rows
                            })
                            all_tables.append({
                                "title": f"Table on Page {page_num + 1}",
                                "page": page_num + 1,
                                "headers": headers,
                                "rows": rows
                            })
            except Exception as e:
                logger.debug(f"Table detection skipped on page {page_num + 1}: {e}")

            # 2. Check if page contains sufficient digital text (at least 40 non-whitespace characters)
            if len(text.replace(" ", "").replace("\n", "")) >= 40:
                used_native = True
                page_text = text
                extraction_mode = "native_text"
            else:
                # Scanned or image-based page -> render to image and run OCR
                logger.info(f"Page {page_num + 1} appears to be scanned or contains minimal digital text. Running OCR...")
                used_ocr = True
                extraction_mode = "scanned_ocr"
                try:
                    pix = page.get_pixmap(dpi=settings.OCR_DPI)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    if self.ocr_engine.is_available():
                        page_text = self.ocr_engine.extract_text(img)
                    else:
                        page_text = text or "[OCR unavailable: Tesseract engine not configured on system]"
                except Exception as ocr_err:
                    logger.warning(f"OCR failed on page {page_num + 1}: {ocr_err}")
                    page_text = text or f"[OCR Error: {ocr_err}]"

            pages_content.append({
                "page": page_num + 1,
                "text": page_text,
                "mode": extraction_mode,
                "tables": page_tables
            })
            raw_text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

        doc.close()

        # Determine overall PDF method
        if used_native and used_ocr:
            pdf_method = "hybrid"
        elif used_ocr:
            pdf_method = "scanned_ocr"
        else:
            pdf_method = "native_text"

        return {
            "filename": filename,
            "file_type": "pdf",
            "page_count": page_count,
            "raw_text": "\n\n".join(raw_text_parts),
            "pages": pages_content,
            "tables": all_tables,
            "pdf_method": pdf_method,
            "ocr_engine": "tesseract" if used_ocr else "none",
            "source_pages": list(range(1, page_count + 1))
        }

    def _extract_image(self, file_path: str, filename: str) -> Dict[str, Any]:
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        try:
            img = Image.open(file_path)
            if self.ocr_engine.is_available():
                extracted_text = self.ocr_engine.extract_text(img)
                ocr_status = "tesseract"
            else:
                extracted_text = "[OCR unavailable: Tesseract engine not configured on system]"
                ocr_status = "none"
        except Exception as e:
            logger.error(f"Failed to process image {filename}: {e}")
            extracted_text = f"[Image extraction error: {e}]"
            ocr_status = "error"

        return {
            "filename": filename,
            "file_type": ext,
            "page_count": 1,
            "raw_text": extracted_text,
            "pages": [{"page": 1, "text": extracted_text, "mode": "scanned_ocr", "tables": []}],
            "tables": [],
            "pdf_method": None,
            "ocr_engine": ocr_status,
            "source_pages": [1]
        }

document_extractor = DocumentExtractor()
