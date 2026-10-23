import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.database import db
from app.routes.documents import router as documents_router
from app.routes.health import router as health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("document_intelligence")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to PostgreSQL database and auto-initialize schema
    logger.info("Starting up Document Intelligence Platform...")
    await db.connect()
    yield
    # Shutdown: Close database pool
    logger.info("Shutting down Document Intelligence Platform...")
    await db.disconnect()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Universal open-source document intelligence, extraction, structured JSON normalization, and retrieval API.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for React Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes under /api/v1
app.include_router(health_router, prefix=settings.API_V1_PREFIX)
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": f"{settings.API_V1_PREFIX}/health",
        "api_endpoints": {
            "extract": f"{settings.API_V1_PREFIX}/documents/extract",
            "list": f"{settings.API_V1_PREFIX}/documents",
            "status": f"{settings.API_V1_PREFIX}/documents/{{document_id}}/status",
            "get_json": f"{settings.API_V1_PREFIX}/documents/{{document_id}}",
            "raw_text": f"{settings.API_V1_PREFIX}/documents/{{document_id}}/raw",
            "download": f"{settings.API_V1_PREFIX}/documents/{{document_id}}/download",
            "rag_query": f"{settings.API_V1_PREFIX}/documents/{{document_id}}/query"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "error": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
