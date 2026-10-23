import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a world-class Universal Document Intelligence Engine.
Your task is to analyze document text and layout, identify its document type, and extract all structured information into the exact JSON schema provided below.

RULES:
1. Do NOT assume every document is an invoice. Classify dynamically: tax_invoice, delivery_challan, purchase_order, bill_receipt, salary_slip, bank_statement, contract_agreement, business_report, letter_form, resume, technical_doc, educational_doc, or generic_document.
2. For financial documents (invoices, challans, bills, bank statements): extract all line items, quantities, rates, tax breakdowns (CGST, SGST, IGST), grand totals, GSTINs, supplier/client details into both standard content and custom_fields.
3. For contracts: extract parties, clauses, dates, obligations, terms.
4. For bank statements: extract account details, transaction table (dates, descriptions, debit, credit, balances).
5. For reports/letters: extract sections, paragraphs, entities, key points.
6. Preserve unknown or unexpected fields in custom_fields. Never discard real information.
7. Return null or empty lists for absent fields. DO NOT invent facts.
8. Output ONLY pure, raw JSON. Do NOT wrap in markdown fences (no ```json).

REQUIRED JSON SCHEMA:
{
  "document_type": "detected_document_type",
  "metadata": {
    "title": "Document Title or null",
    "language": "en",
    "dates": ["date1", "date2"],
    "names": ["name1", "name2"],
    "addresses": ["address1"]
  },
  "content": {
    "sections": [{"title": "Section Title", "content": "Text"}],
    "paragraphs": ["paragraph text"],
    "tables": [
      {
        "title": "Table Title",
        "page": 1,
        "headers": ["col1", "col2"],
        "rows": [{"col1": "val1", "col2": "val2"}]
      }
    ],
    "key_value_pairs": [
      {"key": "Original Label", "value": "Extracted Value", "page": 1}
    ],
    "entities": [
      {"name": "Entity Name", "type": "ORGANIZATION/PERSON/GPE/ID", "page": 1}
    ],
    "custom_fields": {
      "key": "value"
    }
  }
}
"""

class UniversalLLMExtractor:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.base_url = settings.LLM_BASE_URL
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL

    async def extract_structured_json(
        self,
        raw_text: str,
        detected_tables: List[Dict[str, Any]],
        filename: str
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Attempts structured extraction using open-weight LLM inference (e.g. Qwen on Groq/OpenRouter/Ollama).
        Gracefully falls back to heuristic OCR-only extractor if LLM is unconfigured or unavailable.
        Returns: (extracted_data_dict, extraction_metadata_dict)
        """
        llm_metadata = {
            "llm_provider": "none",
            "model": None,
            "extraction_mode": "ocr_only"
        }

        # Check if LLM endpoint is configured
        if self.base_url and self.api_key and self.provider != "none":
            try:
                logger.info(f"Invoking hosted open-weight LLM ({self.model}) via {self.base_url}...")
                llm_response = await self._call_llm(raw_text, detected_tables, filename)
                llm_metadata["llm_provider"] = self.provider
                llm_metadata["model"] = self.model
                llm_metadata["extraction_mode"] = "llm_enhanced"
                return llm_response, llm_metadata
            except Exception as e:
                logger.warning(f"LLM extraction failed or timed out: {e}. Falling back to heuristic OCR extraction.")

        # Fallback to high-accuracy heuristic extraction
        logger.info("Executing Rule-Based / Heuristic Universal Extractor...")
        extracted_data = self._heuristic_extraction(raw_text, detected_tables, filename)
        return extracted_data, llm_metadata

    async def _call_llm(
        self,
        raw_text: str,
        detected_tables: List[Dict[str, Any]],
        filename: str
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Provide document context with table structure if available
        user_prompt = f"Filename: {filename}\n\nDocument Text:\n{raw_text[:25000]}"
        if detected_tables:
            user_prompt += f"\n\nPre-extracted Tables:\n{json.dumps(detected_tables[:5])}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # Clean markdown codeblocks if model returned them
            clean_content = re.sub(r"^```json\s*", "", content.strip(), flags=re.MULTILINE)
            clean_content = re.sub(r"^```\s*", "", clean_content, flags=re.MULTILINE)
            clean_content = re.sub(r"```$", "", clean_content.strip())
            return json.loads(clean_content)

    def _heuristic_extraction(
        self,
        raw_text: str,
        detected_tables: List[Dict[str, Any]],
        filename: str
    ) -> Dict[str, Any]:
        """
        Sophisticated rule-based extraction that extracts fields, classifies documents,
        detects key-value pairs, dates, amounts, GSTINs, addresses, and tables.
        """
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        lower_text = raw_text.lower()

        # 1. Document Type Classification
        doc_type = "generic_document"
        if "delivery challan" in lower_text or "challan no" in lower_text:
            doc_type = "delivery_challan"
        elif "tax invoice" in lower_text or "gst invoice" in lower_text or "invoice no" in lower_text:
            doc_type = "tax_invoice"
        elif "purchase order" in lower_text or "p.o. no" in lower_text:
            doc_type = "purchase_order"
        elif "bank statement" in lower_text or "account statement" in lower_text:
            doc_type = "bank_statement"
        elif "salary slip" in lower_text or "payslip" in lower_text:
            doc_type = "salary_slip"
        elif "agreement" in lower_text or "contract" in lower_text or "memorandum" in lower_text:
            doc_type = "contract_agreement"
        elif "resume" in lower_text or "curriculum vitae" in lower_text:
            doc_type = "resume"
        elif "bill" in lower_text or "receipt" in lower_text:
            doc_type = "bill_receipt"

        # 2. Extract Dates
        date_pattern = r"\b(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b"
        dates = list(set(re.findall(date_pattern, raw_text, flags=re.IGNORECASE)))

        # 3. Extract GSTIN / Tax IDs (Indian GSTIN format: 2 digits, 5 letters, 4 digits, 1 letter, 1 char, Z, 1 char)
        gstin_pattern = r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b"
        gstins = list(set(re.findall(gstin_pattern, raw_text)))

        # 4. Extract Key-Value Pairs
        kv_pairs = []
        custom_fields: Dict[str, Any] = {}
        kv_pattern = r"^([^:\-\t]{2,35})[:\-\t]\s*(.+)$"

        for line in lines:
            match = re.match(kv_pattern, line)
            if match:
                k = match.group(1).strip()
                v = match.group(2).strip()
                # Exclude noisy labels
                if len(k) > 1 and len(v) > 0 and not k.lower().startswith("page"):
                    kv_pairs.append({"key": k, "value": v, "page": 1})
                    # Normalize key for custom_fields
                    norm_k = re.sub(r"[^a-zA-Z0-9]+", "_", k.lower()).strip("_")
                    if norm_k and norm_k not in custom_fields:
                        custom_fields[norm_k] = v

        # 5. Extract Financial Totals and Specific Fields
        if gstins:
            custom_fields["gstins"] = gstins
            if len(gstins) >= 1:
                custom_fields["supplier_gstin"] = gstins[0]
            if len(gstins) >= 2:
                custom_fields["client_gstin"] = gstins[1]

        # Extract Grand Total / Total Amount
        total_patterns = [
            r"(?:grand\s+total|total\s+amount|net\s+amount|total\s+payable|balance\s+due)[:\s₹Rs\.]*([\d,]+(?:\.\d{2})?)",
            r"(?:cgst)[:\s₹Rs\.]*([\d,]+(?:\.\d{2})?)",
            r"(?:sgst)[:\s₹Rs\.]*([\d,]+(?:\.\d{2})?)",
            r"(?:assessable\s+value|taxable\s+value)[:\s₹Rs\.]*([\d,]+(?:\.\d{2})?)"
        ]
        for pat in total_patterns:
            m = re.search(pat, raw_text, flags=re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", "")
                if "grand" in pat or "total" in pat:
                    custom_fields["grand_total"] = float(val) if "." in val else int(val)
                elif "cgst" in pat:
                    custom_fields["cgst"] = float(val) if "." in val else int(val)
                elif "sgst" in pat:
                    custom_fields["sgst"] = float(val) if "." in val else int(val)
                elif "assessable" in pat or "taxable" in pat:
                    custom_fields["taxable_value"] = float(val) if "." in val else int(val)

        # 6. Extract Tables
        tables = detected_tables.copy()
        
        # If no PyMuPDF tables found, check for plain-text tabular lines
        if not tables:
            tabular_lines = [l for l in lines if "\t" in l or "  " in l]
            if len(tabular_lines) >= 3:
                first_row = [c.strip() for c in re.split(r"\t+|\s{2,}", tabular_lines[0]) if c.strip()]
                if len(first_row) >= 2:
                    rows = []
                    for row_line in tabular_lines[1:15]:
                        cells = [c.strip() for c in re.split(r"\t+|\s{2,}", row_line) if c.strip()]
                        if len(cells) == len(first_row):
                            rows.append(dict(zip(first_row, cells)))
                    if rows:
                        tables.append({
                            "title": "Extracted Table",
                            "page": 1,
                            "headers": first_row,
                            "rows": rows
                        })

        # 7. Identify Names & Addresses
        names = []
        addresses = []
        for idx, line in enumerate(lines[:30]):
            l_lower = line.lower()
            if any(term in l_lower for term in ["m/s", "supplier", "client", "buyer", "sold to", "name"]):
                parts = line.split(":")
                candidate = parts[1].strip() if len(parts) > 1 else line
                if candidate and candidate not in names and len(candidate) < 60:
                    names.append(candidate)
            if any(term in l_lower for term in ["address", "site address", "delivery address", "billed to", "shipped to"]):
                parts = line.split(":")
                candidate = parts[1].strip() if len(parts) > 1 else ""
                # Grab subsequent lines for address
                if idx + 1 < len(lines) and len(lines[idx + 1]) < 100:
                    candidate = f"{candidate} {lines[idx + 1]}".strip()
                if candidate:
                    addresses.append(candidate)

        # 8. Paragraphs & Sections
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 30][:10]

        return {
            "document_type": doc_type,
            "metadata": {
                "title": filename.rsplit(".", 1)[0].replace("_", " ").title(),
                "language": "en",
                "dates": dates,
                "names": names[:5],
                "addresses": addresses[:3]
            },
            "content": {
                "sections": [{"title": "General Content", "content": "\n".join(lines[:15])}],
                "paragraphs": paragraphs,
                "tables": tables,
                "key_value_pairs": kv_pairs[:50],
                "entities": [{"name": g, "type": "GSTIN", "page": 1} for g in gstins],
                "custom_fields": custom_fields
            }
        }

universal_llm_extractor = UniversalLLMExtractor()
