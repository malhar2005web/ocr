from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentSource(BaseModel):
    filename: str
    file_type: str
    page_count: int = 1

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    language: Optional[str] = None
    dates: List[str] = []
    names: List[str] = []
    addresses: List[str] = []

class KeyValuePair(BaseModel):
    key: str
    value: Any
    confidence: Optional[float] = None
    page: Optional[int] = None

class TableRow(BaseModel):
    cells: Dict[str, Any]

class DocumentTable(BaseModel):
    title: Optional[str] = None
    page: Optional[int] = None
    headers: List[str] = []
    rows: List[Dict[str, Any]] = []

class DocumentContent(BaseModel):
    sections: List[Dict[str, Any]] = []
    paragraphs: List[str] = []
    tables: List[Dict[str, Any]] = []
    key_value_pairs: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    custom_fields: Dict[str, Any] = {}

class ExtractionMetadata(BaseModel):
    pdf_method: Optional[str] = None  # native_text, scanned_ocr, hybrid
    ocr_engine: Optional[str] = None  # tesseract, none
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    source_pages: List[int] = []

class ValidationIssue(BaseModel):
    field: Optional[str] = None
    issue_type: str
    message: str
    severity: str = "warning"  # info, warning, error

class ValidationResult(BaseModel):
    status: str = "valid"  # valid, review_needed, calculation_mismatch, invalid
    issues: List[ValidationIssue] = []

class UniversalDocumentJSON(BaseModel):
    document_id: str
    schema_version: str = "1.0"
    document_type: str = "unclassified"
    source: DocumentSource
    metadata: DocumentMetadata
    content: DocumentContent
    raw_text: Optional[str] = None
    extraction_metadata: ExtractionMetadata
    validation: ValidationResult

class DocumentExtractResponse(BaseModel):
    document_id: str
    status: str
    message: str

class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    document_type: Optional[str] = None
    filename: str
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DocumentListItem(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    document_type: str
    status: str
    page_count: int
    created_at: Optional[datetime] = None
    error_message: Optional[str] = None

class DocumentListResponse(BaseModel):
    total: int
    page: int
    limit: int
    documents: List[Dict[str, Any]]

class RAGQueryRequest(BaseModel):
    query: str
    document_id: Optional[str] = None

class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Dict[str, Any]] = []
