import os
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

client = TestClient(app)

def create_sample_pdf_bytes():
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 60), "TAX INVOICE\nInvoice No: INV-2026-999\nDate: 23/10/2026\nGrand Total: 45000.00", fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data
    assert "ocr_engine" in data
    assert "llm_provider" in data
    print("PASS: test_health_endpoint")

def test_extract_and_retrieve_flow():
    pdf_bytes = create_sample_pdf_bytes()
    files = {"file": ("test_invoice_api.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    
    # 1. POST /api/v1/documents/extract
    response = client.post("/api/v1/documents/extract", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    doc_id = data["document_id"]
    assert data["status"] in ["completed", "review_needed", "valid"]
    
    # 2. GET /api/v1/documents/{document_id}/status
    status_resp = client.get(f"/api/v1/documents/{doc_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["document_id"] == doc_id
    assert status_data["filename"] == "test_invoice_api.pdf"

    # 3. GET /api/v1/documents/{document_id}
    json_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert json_resp.status_code == 200
    doc_json = json_resp.json()
    assert doc_json["document_id"] == doc_id
    assert doc_json["schema_version"] == "1.0"
    assert "content" in doc_json
    assert "validation" in doc_json
    assert doc_json["document_type"] == "tax_invoice"

    # 4. GET /api/v1/documents (List and Search)
    list_resp = client.get("/api/v1/documents?documentType=tax_invoice&page=1&limit=10")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(d["id"] == doc_id for d in list_data["documents"])

    # 5. GET /api/v1/documents/{document_id}/raw
    raw_resp = client.get(f"/api/v1/documents/{doc_id}/raw")
    assert raw_resp.status_code == 200
    raw_data = raw_resp.json()
    assert "TAX INVOICE" in raw_data["raw_text"]

    # 6. GET /api/v1/documents/{document_id}/download
    dl_resp = client.get(f"/api/v1/documents/{doc_id}/download")
    assert dl_resp.status_code == 200
    assert len(dl_resp.content) > 0

    # 7. POST /api/v1/documents/{document_id}/query (RAG QA)
    rag_resp = client.post(
        f"/api/v1/documents/{doc_id}/query",
        json={"query": "What is the grand total amount?"}
    )
    assert rag_resp.status_code == 200
    rag_data = rag_resp.json()
    assert "answer" in rag_data
    assert len(rag_data["citations"]) > 0
    print("PASS: test_extract_and_retrieve_flow")

if __name__ == "__main__":
    test_health_endpoint()
    test_extract_and_retrieve_flow()
    print("\nALL API ENDPOINT TESTS PASSED SUCCESSFULLY!")
