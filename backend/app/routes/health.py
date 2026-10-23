import os
from fastapi import APIRouter
from app.core.config import settings
from app.db.database import db
from app.services.ocr.engine import TesseractOCREngine

router = APIRouter(tags=["Health"])
ocr_engine = TesseractOCREngine(tesseract_cmd=settings.TESSERACT_CMD)

@router.get("/health")
async def health_check():
    """Returns platform system health, storage, OCR status, and LLM configuration."""
    # Check upload directory write permissions
    storage_ok = os.path.exists(settings.UPLOAD_DIR) and os.access(settings.UPLOAD_DIR, os.W_OK)
    
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": {
            "type": "PostgreSQL",
            "connected": db.is_connected,
            "host": settings.POSTGRES_HOST,
            "database": settings.POSTGRES_DB
        },
        "ocr_engine": {
            "name": "Tesseract OCR",
            "available": ocr_engine.is_available(),
            "binary_path": ocr_engine.tesseract_cmd
        },
        "llm_provider": {
            "provider": settings.LLM_PROVIDER,
            "configured": bool(settings.LLM_BASE_URL and settings.LLM_API_KEY),
            "model": settings.LLM_MODEL
        },
        "storage": {
            "upload_dir": settings.UPLOAD_DIR,
            "writeable": storage_ok,
            "max_file_size_mb": settings.MAX_FILE_SIZE_MB
        },
        "rag_enabled": settings.ENABLE_RAG
    }
