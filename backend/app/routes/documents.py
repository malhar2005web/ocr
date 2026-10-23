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
