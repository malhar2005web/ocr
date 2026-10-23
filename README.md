# Universal Document Intelligence & Extraction Platform

An open-source, deployable web application that extracts information from virtually any PDF document or image (tax invoices, delivery challans, purchase orders, contracts, bank statements, bills, reports, forms, and photographs), converts it into normalized structured JSON, persists results in PostgreSQL, and exposes REST APIs and interactive RAG semantic retrieval.

---

## 🌟 Key Features

- **Universal Document Ingestion**: Supports Native PDFs, Scanned PDFs, JPG, JPEG, and PNG.
- **Smart Dual-Engine Extraction**:
  - **PyMuPDF (`fitz`)**: Lightning-fast native text and tabular extraction with font heuristics and page boundaries.
  - **Tesseract OCR with OpenCV Preprocessing**: Automatic grayscale conversion, deskewing, noise reduction, and adaptive thresholding for scanned documents and camera photos.
  - **No Unnecessary OCR**: Intelligently skips OCR when usable digital text is present.
- **Universal JSON Schema**: Consistent outer envelope (`document_id`, `schema_version`, `document_type`, `source`, `metadata`, `content`, `raw_text`, `extraction_metadata`, `validation`).
- **Configurable Open-Weight LLM & Heuristic Fallback**:
  - Hosted OpenAI-compatible inference with Qwen 2.5 (72B / 14B) or Llama 3 via Groq, OpenRouter, Ollama, or vLLM.
  - High-accuracy zero-config **OCR-Only Heuristic Mode** that extracts key-values, GSTINs, totals, taxes, and tables without requiring any paid API.
- **Automated Validation & Mathematical Verification**:
  - Flags calculation discrepancies (e.g. `Assessable Value + CGST + SGST != Grand Total`) without silently altering extracted numbers.
- **PostgreSQL Persistence**:
  - Uses relational columns for rapid filtering and `JSONB` with GIN indexing for flexible document schemas.
  - Graceful fallback store ensures server availability even during initial database provisioning.
- **Interactive React (JSX) Frontend**:
  - Executive Dashboard with ingestion KPIs and health checks.
  - Multi-file Drag & Drop upload with live progress tracking.
  - Split-screen Document Inspector (original file preview alongside extracted JSON and tables).
  - Search & Explorer with document type, date, status, and nested JSON filters.
  - Natural Language Document QA (RAG Assistant) citing source pages.
  - Built-in REST API Documentation and cURL code generator.

---

## 🏛️ System Architecture

```
                       [React Frontend (Vite / JSX)]
                                    │
                               REST APIs (/api/v1)
                                    ▼
                         [FastAPI Backend Engine]
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ▼                                                       ▼
[PyMuPDF (fitz)] (Digital PDF)                   [Tesseract OCR + OpenCV] (Scanned/Image)
        │                                                       │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                    [Document Structure Organizer]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [Open-Weight LLM] (Qwen / Llama)      [Heuristic Extractor] (OCR-Only Fallback)
                  └─────────────────┬─────────────────┘
                                    ▼
                       [Validation & Normalizer]
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        [PostgreSQL + JSONB]                  [RAG Semantic Chunker]
```

---

## 🚀 Quickstart & Installation (Normal Setup without Docker)

### 1. Prerequisites
- Python 3.10+
- Node.js v18+ and npm
- PostgreSQL 14+ (already installed on port 5432)
- Tesseract OCR (Optional for scanned PDFs and photos; native digital PDFs work out of the box)

### 2. Backend Setup

```bash
cd backend

# Install Python dependencies
python -m pip install -r requirements.txt

# Create .env from template
cp ../.env.example .env
```

Edit `.env` to configure your PostgreSQL credentials:
```ini
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=doc_intelligence

# Optional Hosted LLM (Leave empty for high-accuracy OCR-only mode)
LLM_PROVIDER=open_weight
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_or_openrouter_api_key
LLM_MODEL=qwen-2.5-72b-instruct
```

Start the FastAPI backend:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API and Swagger documentation will be available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup

```bash
cd frontend

# Install frontend dependencies
npm install

# Start Vite React development server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 📡 REST API Reference

All endpoints use the `/api/v1` prefix.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/extract` | Upload & extract PDF/image. Multipart form-data (`file`, `async_mode`). |
| `GET` | `/api/v1/documents/{document_id}` | Retrieve complete normalized Universal JSON. |
| `GET` | `/api/v1/documents/{document_id}/status` | Retrieve processing status and progress. |
| `GET` | `/api/v1/documents` | List & filter documents by type, filename, date, status, or JSONB content. |
| `GET` | `/api/v1/documents/{document_id}/raw` | Retrieve extracted raw text and extraction metadata. |
| `GET` | `/api/v1/documents/{document_id}/download` | Download original uploaded document. |
| `POST` | `/api/v1/documents/{document_id}/query` | Ask natural language questions with source citations (RAG). |
| `GET` | `/api/v1/health` | Check service health, PostgreSQL, and OCR engine status. |

### cURL Examples

#### 1. Ingest & Extract a Document:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/extract" \
  -F "file=@/path/to/sample_invoice.pdf" \
  -F "async_mode=false"
```

#### 2. Query Documents with Multi-Parameter Filters:
```bash
curl -X GET "http://localhost:8000/api/v1/documents?documentType=tax_invoice&status=completed&limit=10"
```

#### 3. Ask Document Content (RAG):
```bash
curl -X POST "http://localhost:8000/api/v1/documents/<DOCUMENT_ID>/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the payment terms mentioned in this document?"}'
```

---

## 🧪 Running Automated Tests

Run the complete extraction and API test suites:

```bash
# 1. Pipeline Tests (PyMuPDF, Invoices, Contracts, Tables, Calculations, Images)
$env:PYTHONPATH="d:\Desktop\ocr\backend"; python d:\Desktop\ocr\backend\tests\test_extraction_pipeline.py

# 2. End-to-End API Integration Tests
$env:PYTHONPATH="d:\Desktop\ocr\backend"; python d:\Desktop\ocr\backend\tests\test_api_endpoints.py
```

---

## 📋 Universal JSON Output Schema

```json
{
  "document_id": "c86e245a-8b82-4f25-83e9-74d1a499313b",
  "schema_version": "1.0",
  "document_type": "tax_invoice",
  "source": {
    "filename": "invoice_2800.pdf",
    "file_type": "pdf",
    "page_count": 1
  },
  "metadata": {
    "title": "Tax Invoice",
    "language": "en",
    "dates": ["18/09/2026"],
    "names": ["SAI SIYA INFRA", "Ananya Infrastructure"],
    "addresses": ["Plot 42, Sector 18, Navi Mumbai, MH"]
  },
  "content": {
    "sections": [...],
    "paragraphs": [...],
    "tables": [
      {
        "title": "Item Details",
        "page": 1,
        "headers": ["Item Description", "HSN Code", "Qty", "Rate", "Assessable Value"],
        "rows": [
          { "Item Description": "M30 Concrete Mix", "HSN Code": "38245010", "Qty": "4.50 CU.MTR", "Rate": "5500.00", "Assessable Value": "24750.00" }
        ]
      }
    ],
    "key_value_pairs": [
      { "key": "Invoice No", "value": "2800", "page": 1 },
      { "key": "Vehicle No", "value": "MH-12-AB-1234", "page": 1 }
    ],
    "entities": [
      { "name": "27ANKPC6900E2ZA", "type": "GSTIN", "page": 1 }
    ],
    "custom_fields": {
      "supplier_name": "SAI SIYA INFRA",
      "supplier_gstin": "27ANKPC6900E2ZA",
      "client_name": "Ananya Infrastructure",
      "client_gstin": "27ABRFA3679D1Z4",
      "taxable_value": 24750.0,
      "cgst": 2227.5,
      "sgst": 2227.5,
      "grand_total": 29205.0
    }
  },
  "raw_text": "...",
  "extraction_metadata": {
    "pdf_method": "native_text",
    "ocr_engine": "none",
    "llm_provider": "open_weight",
    "model": "qwen-2.5-72b-instruct",
    "source_pages": [1]
  },
  "validation": {
    "status": "valid",
    "issues": []
  }
}
```
