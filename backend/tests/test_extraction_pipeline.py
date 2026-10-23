import os
import io
import pytest
import fitz
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.services.extractor import document_extractor
from app.services.llm_extractor import universal_llm_extractor
from app.services.validator import document_validator
from app.services.rag_service import rag_service

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
os.makedirs(TEST_DIR, exist_ok=True)

def create_sample_digital_pdf(path: str, content: str, title: str = "Test Doc"):
    """Creates a native digital PDF using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 72), f"{title}\n\n{content}", fontsize=11)
    doc.save(path)
    doc.close()

def create_sample_invoice_pdf(path: str):
    """Creates a sample GST invoice PDF with structured tables and calculations."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    invoice_text = """
    TAX INVOICE
    Supplier Name: SAI SIYA INFRA
    Supplier GSTIN: 27ANKPC6900E2ZA
    Address: Plot 42, Sector 18, Navi Mumbai, MH
    
    Client Name: Ananya Infrastructure
    Client GSTIN: 27ABRFA3679D1Z4
    Site Address: Site 9B, Pune Road, Maharashtra
    
    Invoice No: 2800
    Date: 18/09/2026
    Vehicle No: MH-12-AB-1234
    
    Item Description | HSN Code | Qty | Rate | Assessable Value
    M30 Concrete Mix | 38245010 | 4.50 CU.MTR | 5500.00 | 24750.00
    
    Taxable Value: 24750.00
    CGST: 2227.50
    SGST: 2227.50
    Grand Total: 29205.00
    """
    page.insert_text((50, 60), invoice_text, fontsize=10)
    doc.save(path)
    doc.close()

def create_sample_contract_pdf(path: str):
    """Creates a multi-page contract document."""
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1_text = """
    SERVICE LEVEL AGREEMENT
    
    This Agreement is entered into on 01/04/2026 by and between:
    Provider: Apex Cloud Technologies Ltd
    Address: 100 Cyber City, Hyderabad
    
    Client: Global Retail Corp
    Address: 500 Market St, Bangalore
    
    1. Scope of Services
    The Provider shall supply 99.9% uptime for cloud hosting and continuous data monitoring.
    """
    p1.insert_text((50, 60), p1_text, fontsize=10)
    
    # Page 2
    p2 = doc.new_page(width=595, height=842)
    p2_text = """
    2. Payment Terms
    All invoices shall be paid within 30 days of receipt.
    Interest on late payments shall accrue at 1.5% per month.
    
    3. Termination
    Either party may terminate this agreement with 60 days written notice.
    """
    p2.insert_text((50, 60), p2_text, fontsize=10)
    doc.save(path)
    doc.close()

def create_sample_image(path: str):
    """Creates a synthetic test image."""
    img = Image.new("RGB", (600, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "DELIVERY CHALLAN\nChallan No: CH-9081\nDate: 20/10/2026\nGrand Total: 15400.00", fill=(0, 0, 0))
    img.save(path)

def test_native_pdf_extraction():
    pdf_path = os.path.join(TEST_DIR, "test_digital.pdf")
    create_sample_digital_pdf(pdf_path, "This is a clean digitally generated PDF containing test text.", "Digital Report")
    
    result = document_extractor.extract_document(pdf_path, "test_digital.pdf")
    assert result["pdf_method"] == "native_text"
    assert "Digital Report" in result["raw_text"]
    assert result["page_count"] == 1
    print("PASS: test_native_pdf_extraction")

def test_invoice_extraction_and_validation():
    inv_path = os.path.join(TEST_DIR, "test_invoice.pdf")
    create_sample_invoice_pdf(inv_path)
    
    result = document_extractor.extract_document(inv_path, "test_invoice.pdf")
    assert result["page_count"] == 1
    
    # Run heuristic extractor
    structured = universal_llm_extractor._heuristic_extraction(
        result["raw_text"],
        result["tables"],
        "test_invoice.pdf"
    )
    assert structured["document_type"] == "tax_invoice"
    custom = structured["content"]["custom_fields"]
    assert "27ANKPC6900E2ZA" in custom.get("gstins", []) or "27ABRFA3679D1Z4" in custom.get("gstins", [])
    assert custom.get("grand_total") == 29205.00
    assert custom.get("cgst") == 2227.50
    assert custom.get("sgst") == 2227.50
    
    # Run validation
    norm_doc = {
        "document_type": structured["document_type"],
        "metadata": structured["metadata"],
        "content": structured["content"]
    }
    val_res = document_validator.validate(norm_doc)
    assert val_res["status"] == "valid"
    print("PASS: test_invoice_extraction_and_validation")

def test_calculation_mismatch_detection():
    # Construct an invoice where taxable + taxes != grand_total
    bad_doc = {
        "document_type": "tax_invoice",
        "metadata": {"dates": ["18/09/2026"]},
        "content": {
            "custom_fields": {
                "taxable_value": 20000.00,
                "cgst": 1000.00,
                "sgst": 1000.00,
                "grand_total": 50000.00  # Obvious mismatch: 22000 != 50000
            }
        }
    }
    val_res = document_validator.validate(bad_doc)
    assert val_res["status"] == "calculation_mismatch"
    assert len(val_res["issues"]) > 0
    assert val_res["issues"][0]["issue_type"] == "calculation_mismatch"
    print("PASS: test_calculation_mismatch_detection")

def test_multipage_contract_and_rag():
    contract_path = os.path.join(TEST_DIR, "test_contract.pdf")
    create_sample_contract_pdf(contract_path)
    
    res = document_extractor.extract_document(contract_path, "test_contract.pdf")
    assert res["page_count"] == 2
    
    # Chunker
    chunks = rag_service.chunk_document(res["raw_text"], res["pages"])
    assert len(chunks) >= 2
    
    # Semantic Search for "Payment Terms"
    search_hits = rag_service.search_chunks("payment terms", chunks)
    assert len(search_hits) > 0
    assert search_hits[0]["page"] == 2
    assert "30 days" in search_hits[0]["content"]
    print("PASS: test_multipage_contract_and_rag")

def test_image_processing():
    img_path = os.path.join(TEST_DIR, "test_challan.png")
    create_sample_image(img_path)
    
    res = document_extractor.extract_document(img_path, "test_challan.png")
    assert res["file_type"] == "png"
    assert res["page_count"] == 1
    print("PASS: test_image_processing")

if __name__ == "__main__":
    test_native_pdf_extraction()
    test_invoice_extraction_and_validation()
    test_calculation_mismatch_detection()
    test_multipage_contract_and_rag()
    test_image_processing()
    print("\nALL TEST PIPELINE SUITES PASSED EXCELLENTLY!")
