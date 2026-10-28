import os
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import UploadFile

from app.core.config import settings
from app.db.database import db
from app.services.extractor import document_extractor
from app.services.llm_extractor import universal_llm_extractor
from app.services.validator import document_validator
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

class DocumentController:
    async def process_document(self, file: UploadFile, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Coordinates full end-to-end pipeline:
        File validation -> Storage -> Extraction -> LLM Structuring -> Validation -> Persistence.
        """
        if not doc_id:
            doc_id = str(uuid.uuid4())

        # 1. Sanitize filename & extension
        raw_filename = file.filename or "uploaded_document"
        sanitized_filename = os.path.basename(raw_filename).replace(" ", "_")
        ext = os.path.splitext(sanitized_filename)[1].lower().lstrip(".")

        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(f"File extension '{ext}' not permitted. Allowed: {settings.ALLOWED_EXTENSIONS}")

        # 2. Save file to persistent storage
        file_save_name = f"{doc_id}_{sanitized_filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, file_save_name)

        content = await file.read()
        file_size = len(content)

        max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(f"File size {file_size / (1024*1024):.1f}MB exceeds limit of {settings.MAX_FILE_SIZE_MB}MB.")

        with open(file_path, "wb") as f:
            f.write(content)

        # 3. Create initial document record with 'processing' status
        doc_record = {
            "id": doc_id,
            "filename": sanitized_filename,
            "file_type": ext,
            "file_path": file_path,
            "file_size": file_size,
            "document_type": "unclassified",
            "status": "processing",
            "page_count": 1,
            "raw_text": None,
            "normalized_json": None,
            "metadata_json": None,
            "error_message": None
        }
        await db.save_document(doc_record)
        await db.log_audit(doc_id, "UPLOAD_STARTED", {"filename": sanitized_filename, "size": file_size})

        try:
            # 4. Text and Layout Extraction (PyMuPDF native + Tesseract OCR)
            extraction_result = document_extractor.extract_document(file_path, sanitized_filename)
            raw_text = extraction_result["raw_text"]
            page_count = extraction_result["page_count"]
            detected_tables = extraction_result["tables"]

            layout_regions = extraction_result.get("layout_regions")

            # 5. Structured JSON Extraction (Open-weight LLM or Heuristic Fallback)
            extracted_data, llm_meta = await universal_llm_extractor.extract_structured_json(
                raw_text=raw_text,
                detected_tables=detected_tables,
                filename=sanitized_filename,
                layout_regions=layout_regions
            )

            # 6. Compose Universal JSON Output
            detected_type = extracted_data.get("document_type", "generic_document")
            doc_metadata = extracted_data.get("metadata", {})
            doc_content = extracted_data.get("content", {})

            extraction_metadata = {
                "pdf_method": extraction_result["pdf_method"],
                "ocr_engine": extraction_result["ocr_engine"],
                "llm_provider": llm_meta["llm_provider"],
                "model": llm_meta["model"],
                "extraction_mode": llm_meta["extraction_mode"],
                "cross_verification_status": llm_meta.get("cross_verification_status", "verified"),
                "source_pages": extraction_result["source_pages"]
            }

            candidate_json = {
                "document_id": doc_id,
                "schema_version": "1.0",
                "document_type": detected_type,
                "source": {
                    "filename": sanitized_filename,
                    "file_type": ext,
                    "page_count": page_count
                },
                "metadata": doc_metadata,
                "content": doc_content,
                "raw_text": raw_text,
                "extraction_metadata": extraction_metadata,
                "validation": {"status": "pending_review", "issues": []}
            }

            # 7. Validation
            validation_res = document_validator.validate(candidate_json)
            candidate_json["validation"] = validation_res

            # Determine final status
            final_status = "completed"
            if validation_res["status"] in ["review_needed", "calculation_mismatch"]:
                final_status = "review_needed"

            # 8. Store RAG Chunks
            chunks = rag_service.chunk_document(raw_text, extraction_result["pages"])
            candidate_json["_rag_chunks"] = chunks

            # 9. Update DB with complete results
            doc_record.update({
                "document_type": detected_type,
                "status": final_status,
                "page_count": page_count,
                "raw_text": raw_text,
                "normalized_json": candidate_json,
                "metadata_json": doc_metadata,
                "error_message": None
            })
            await db.save_document(doc_record)
            await db.log_audit(doc_id, "PROCESSING_COMPLETED", {"status": final_status, "document_type": detected_type})

            return candidate_json

        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
            doc_record["status"] = "failed"
            doc_record["error_message"] = str(e)
            await db.save_document(doc_record)
            await db.log_audit(doc_id, "PROCESSING_FAILED", {"error": str(e)})
            raise

document_controller = DocumentController()
