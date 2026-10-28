import os
import uuid
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.db.database import db
from app.controllers.document_controller import document_controller
from app.services.extractor import document_extractor
from app.services.llm_extractor import universal_llm_extractor
from app.services.validator import document_validator
from app.schemas.document import (
    DocumentExtractResponse,
    DocumentStatusResponse,
    DocumentListResponse,
    RAGQueryRequest,
    RAGQueryResponse
)
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/extract", response_model=DocumentExtractResponse)
async def extract_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_mode: bool = Form(False)
):
    """
    Upload a PDF or image document.
    Extracts text, organizes fields, generates Universal JSON, and persists to PostgreSQL.
    """
    doc_id = str(uuid.uuid4())
    
    if async_mode:
        # Run extraction in background task
        background_tasks.add_task(document_controller.process_document, file, doc_id)
        return DocumentExtractResponse(
            document_id=doc_id,
            status="processing",
            message="Document upload accepted. Processing started in background."
        )
    else:
        # Synchronous execution
        try:
            result = await document_controller.process_document(file, doc_id)
            return DocumentExtractResponse(
                document_id=doc_id,
                status=result.get("validation", {}).get("status", "completed"),
                message="Document processed and extracted successfully."
            )
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Retrieves current processing status, errors, and metadata of a document."""
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentStatusResponse(
        document_id=doc["id"],
        status=doc.get("status", "unknown"),
        document_type=doc.get("document_type"),
        filename=doc.get("filename", ""),
        error_message=doc.get("error_message"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at")
    )

@router.get("/{document_id}")
async def get_document_json(document_id: str):
    """Returns the complete normalized Universal JSON for the requested document."""
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    normalized = doc.get("normalized_json")
    if not normalized:
        # If still processing or failed
        return JSONResponse(
            status_code=202 if doc.get("status") == "processing" else 400,
            content={
                "document_id": doc["id"],
                "status": doc.get("status"),
                "message": doc.get("error_message") or "Document extraction is in progress."
            }
        )
    return normalized

@router.post("/{document_id}/reprocess")
async def reprocess_document(document_id: str):
    """Re-runs OCR extraction, table parsing, and LLM structuring on an existing uploaded file."""
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = doc.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stored document file not found on disk")
    
    # Run full updated extraction pipeline
    extraction_result = document_extractor.extract_document(file_path, doc["filename"])
    raw_text = extraction_result["raw_text"]
    page_count = extraction_result["page_count"]
    detected_tables = extraction_result["tables"]

    layout_regions = extraction_result.get("layout_regions")

    extracted_data, llm_meta = await universal_llm_extractor.extract_structured_json(
        raw_text=raw_text,
        detected_tables=detected_tables,
        filename=doc["filename"],
        layout_regions=layout_regions
    )

    detected_type = extracted_data.get("document_type", "generic_document")
    doc_metadata = extracted_data.get("metadata", {})
    doc_content = extracted_data.get("content", {})

    extraction_metadata = {
        "pdf_method": extraction_result["pdf_method"],
        "ocr_engine": extraction_result["ocr_engine"],
        "llm_provider": llm_meta["llm_provider"],
        "model": llm_meta["model"],
        "extraction_mode": llm_meta["extraction_mode"],
        "source_pages": extraction_result["source_pages"]
    }

    candidate_json = {
        "document_id": document_id,
        "schema_version": "1.0",
        "document_type": detected_type,
        "source": {
            "filename": doc["filename"],
            "file_type": doc.get("file_type", "jpeg"),
            "page_count": page_count
        },
        "metadata": doc_metadata,
        "content": doc_content,
        "raw_text": raw_text,
        "extraction_metadata": extraction_metadata,
        "validation": {"status": "pending_review", "issues": []}
    }

    validation_res = document_validator.validate(candidate_json)
    candidate_json["validation"] = validation_res

    final_status = "completed"
    if validation_res["status"] in ["review_needed", "calculation_mismatch"]:
        final_status = "review_needed"

    chunks = rag_service.chunk_document(raw_text, extraction_result["pages"])
    candidate_json["_rag_chunks"] = chunks

    doc.update({
        "document_type": detected_type,
        "status": final_status,
        "page_count": page_count,
        "raw_text": raw_text,
        "normalized_json": candidate_json,
        "metadata_json": doc_metadata,
        "error_message": None
    })
    await db.save_document(doc)
    await db.log_audit(document_id, "REPROCESSED", {"status": final_status, "document_type": detected_type})
    return candidate_json

@router.post("/{document_id}/approve")
async def approve_document(document_id: str):
    """Manually approves a document under review, resolving warnings and setting status to completed."""
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc["status"] = "completed"
    if doc.get("normalized_json"):
        if "validation" not in doc["normalized_json"]:
            doc["normalized_json"]["validation"] = {}
        doc["normalized_json"]["validation"]["status"] = "completed"
        doc["normalized_json"]["validation"]["issues"] = []
    
    await db.save_document(doc)
    await db.log_audit(document_id, "MANUAL_APPROVAL", {"status": "completed"})
    return doc.get("normalized_json") or {"status": "completed", "document_id": document_id}

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    documentType: Optional[str] = Query(None, description="Filter by document type (tax_invoice, delivery_challan, etc.)"),
    filename: Optional[str] = Query(None, description="Search by filename keyword"),
    status: Optional[str] = Query(None, description="Filter by status (completed, review_needed, failed, processing)"),
    fromDate: Optional[str] = Query(None, description="Filter from ISO date (YYYY-MM-DD)"),
    toDate: Optional[str] = Query(None, description="Filter to ISO date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    sortBy: str = Query("created_at", description="Field to sort by"),
    sortOrder: str = Query("DESC", description="Sort order: ASC or DESC"),
    json_filter: Optional[str] = Query(None, description="JSON string to filter against JSONB normalized_json")
):
    """
    Search and filter processed documents by metadata, dates, status, or nested JSON fields.
    """
    parsed_json_filter = None
    if json_filter:
        try:
            parsed_json_filter = json.loads(json_filter)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for json_filter parameter")

    result = await db.list_documents(
        document_type=documentType,
        filename=filename,
        status=status,
        from_date=fromDate,
        to_date=toDate,
        page=page,
        limit=limit,
        sort_by=sortBy,
        sort_order=sortOrder,
        json_filter=parsed_json_filter
    )
    return result

@router.get("/{document_id}/raw")
async def get_raw_text(document_id: str):
    """Returns the extracted source raw text and permitted extraction metadata."""
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": doc["id"],
        "filename": doc.get("filename"),
        "raw_text": doc.get("raw_text"),
        "page_count": doc.get("page_count", 1),
        "extraction_metadata": (doc.get("normalized_json") or {}).get("extraction_metadata", {})
    }

@router.get("/{document_id}/download")
async def download_original_file(document_id: str):
    """Downloads the original uploaded document."""
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    file_path = doc.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Physical document file not found on disk")
    
    return FileResponse(
        path=file_path,
        filename=doc.get("filename", "document"),
        media_type="application/octet-stream"
    )

@router.post("/{document_id}/query", response_model=RAGQueryResponse)
async def query_document_rag(document_id: str, payload: RAGQueryRequest):
    """
    Ask natural-language questions about this document (RAG module).
    Returns answer citing source page and text excerpts.
    """
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    norm = doc.get("normalized_json") or {}
    chunks = norm.get("_rag_chunks", [])
    
    # If chunks not cached, rebuild on-demand
    if not chunks and doc.get("raw_text"):
        chunks = rag_service.chunk_document(doc["raw_text"], [{"page": 1, "text": doc["raw_text"]}])

    result = await rag_service.answer_question(
        query=payload.query,
        chunks=chunks,
        document_filename=doc.get("filename", "document")
    )
    return result
