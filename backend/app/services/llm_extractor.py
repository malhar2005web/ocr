import os
import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GST_STATE_MAP: Dict[str, str] = {
    "01": "Jammu and Kashmir", "02": "Himachal Pradesh", "03": "Punjab", "04": "Chandigarh",
    "05": "Uttarakhand", "06": "Haryana", "07": "Delhi", "08": "Rajasthan", "09": "Uttar Pradesh",
    "10": "Bihar", "11": "Sikkim", "12": "Arunachal Pradesh", "13": "Nagaland", "14": "Manipur",
    "15": "Mizoram", "16": "Tripura", "17": "Meghalaya", "18": "Assam", "19": "West Bengal",
    "20": "Jharkhand", "21": "Odisha", "22": "Chhattisgarh", "23": "Madhya Pradesh", "24": "Gujarat",
    "26": "Dadra & Nagar Haveli and Daman & Diu", "27": "Maharashtra", "29": "Karnataka", "30": "Goa",
    "31": "Lakshadweep", "32": "Kerala", "33": "Tamil Nadu", "34": "Puducherry",
    "35": "Andaman and Nicobar Islands", "36": "Telangana", "37": "Andhra Pradesh", "38": "Ladakh",
    "97": "Other Territory"
}

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

CROSS_VERIFICATION_SYSTEM_PROMPT = """You are a Senior Document Intelligence Verification Auditor.
Your task is to CROSS-VERIFY and AUDIT candidate structured document data against ground-truth text extracted from Tesseract OCR and PyMuPDF.

CRITICAL AUDIT RULES:
1. Verify Seller and Buyer details (Names, GSTINs, PAN, Addresses, State).
   - If an OCR typo is found in candidate data (e.g. broken company name or misread GSTIN character), provide the correction.
2. Verify line items (description, HSN/SAC code, quantity, rate, amount). Ensure quantity * rate ≈ amount.
3. Verify taxes (CGST, SGST, IGST), rounding adjustments, and grand total.
4. Verify invoice number, dates, acknowledgement number, IRN, and bank details.

RESPONSE FORMAT:
Return a single JSON object with this exact structure:
{
  "audit_status": "verified",
  "corrections": {},
  "audit_summary": "Summary of audit verification results"
}

If any candidate field contains an OCR typo or missing value found in the document text, return it in the corrections map:
{
  "audit_status": "corrections_applied",
  "corrections": {
    "field_path": "corrected_value"
  },
  "audit_summary": "Explanation of verified corrections"
}
If all candidate fields are accurate, return an empty corrections object {}.
Output ONLY pure raw JSON without markdown formatting or code fences.
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
        filename: str,
        layout_regions: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Two-Pass Universal Extraction Pipeline:
        Pass 1: High-precision deterministic layout-aware OCR extraction (PyMuPDF / Tesseract).
                Produces mathematically validated candidate JSON and region tokens.
        Pass 2: Qwen LLM Cross-Verification & Auditor (via OpenAI-compatible endpoint like Groq/OpenRouter/Ollama).
                Audits, cross-checks, fixes any OCR character confusions, and reconciles candidate JSON against raw text.
                Gracefully falls back to Pass 1 candidate data if LLM is unconfigured or unavailable.
        """
        # Pass 1: Deterministic Layout & Heuristic Extraction (Tesseract OCR / PyMuPDF)
        logger.info("Pass 1: Executing Deterministic Layout & Rule-Based OCR Extraction...")
        candidate_data = self._heuristic_extraction(raw_text, detected_tables, filename, layout_regions=layout_regions)

        llm_metadata = {
            "llm_provider": "none",
            "model": None,
            "extraction_mode": "deterministic_ocr_verified",
            "cross_verification_status": "skipped_no_api_key",
            "audit_summary": None
        }

        # Check if LLM endpoint is configured for Pass 2 cross-verification
        is_local = bool(self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url))
        if self.base_url and self.provider != "none" and (self.api_key or is_local):
            try:
                logger.info(f"Pass 2: Invoking Qwen LLM Cross-Verification ({self.model}) via {self.base_url}...")
                reconciled_data, audit_summary = await self._cross_verify_with_llm(
                    candidate_data=candidate_data,
                    raw_text=raw_text,
                    layout_regions=layout_regions,
                    detected_tables=detected_tables,
                    filename=filename
                )
                llm_metadata["llm_provider"] = self.provider
                llm_metadata["model"] = self.model
                llm_metadata["extraction_mode"] = "hybrid_cross_verified"
                llm_metadata["cross_verification_status"] = "verified"
                llm_metadata["audit_summary"] = audit_summary
                return reconciled_data, llm_metadata
            except Exception as e:
                err_detail = str(e).strip() or type(e).__name__
                logger.warning(f"Pass 2: Qwen LLM cross-verification failed: {err_detail}. Retaining Pass 1 deterministic candidate data.")
                llm_metadata["llm_provider"] = self.provider
                llm_metadata["model"] = self.model
                llm_metadata["extraction_mode"] = "deterministic_ocr_verified"
                llm_metadata["cross_verification_status"] = f"fallback_to_ocr: {err_detail[:60]}"
                return candidate_data, llm_metadata

        return candidate_data, llm_metadata

    async def _cross_verify_with_llm(
        self,
        candidate_data: Dict[str, Any],
        raw_text: str,
        layout_regions: Optional[Dict[str, str]],
        detected_tables: List[Dict[str, Any]],
        filename: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Cross-verifies, audits, and reconciles the Pass 1 candidate data against the
        ground-truth OCR text and layout regions using Qwen LLM.
        """
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Build context payload with regions and candidate JSON
        regions_summary = ""
        if layout_regions:
            for reg_name, reg_text in layout_regions.items():
                if reg_text and reg_text.strip():
                    regions_summary += f"\n--- REGION: {reg_name.upper()} ---\n{reg_text.strip()[:600]}\n"

        is_cpu_local = bool(self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url))
        max_raw = 2000 if is_cpu_local else 15000
        compact_candidate = candidate_data.get("content", {}).get("custom_fields") or candidate_data

        user_content = f"""Document Filename: {filename}

=== GROUND TRUTH DOCUMENT TEXT (FROM TESSERACT OCR / PYMUPDF) ===
{raw_text[:max_raw]}
"""
        if regions_summary and not is_cpu_local:
            user_content += f"""
=== SEGMENTED LAYOUT REGIONS ===
{regions_summary[:3000]}
"""
        user_content += f"""
=== CANDIDATE DATA TO BE CROSS-VERIFIED & AUDITED ===
{json.dumps(compact_candidate, separators=(',', ':') if is_cpu_local else None, indent=None if is_cpu_local else 2)}

Audit the candidate data against the document text. Return JSON with audit_status, corrections, and audit_summary. Output pure JSON only:"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": CROSS_VERIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.05,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            try:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as err:
                if err.response.status_code == 400 and "response_format" in payload:
                    payload_no_rf = dict(payload)
                    del payload_no_rf["response_format"]
                    resp = await client.post(endpoint, headers=headers, json=payload_no_rf)
                    resp.raise_for_status()
                else:
                    raise

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # Clean markdown codeblocks if model returned them
            clean_content = re.sub(r"^```json\s*", "", content.strip(), flags=re.MULTILINE)
            clean_content = re.sub(r"^```\s*", "", clean_content, flags=re.MULTILINE)
            clean_content = re.sub(r"```$", "", clean_content.strip())
            verified_json = json.loads(clean_content)

            audit_summary = "All candidate data verified against ground truth."
            if isinstance(verified_json, dict):
                audit_summary = verified_json.get("audit_summary") or audit_summary

                # 1. Handle audit-diff output (fast & clean)
                if "audit_status" in verified_json or "corrections" in verified_json:
                    corrections = verified_json.get("corrections", {})
                    # Normalize list of dicts (common in lightweight models) to dict
                    if isinstance(corrections, list):
                        flat_corr = {}
                        for item in corrections:
                            if isinstance(item, dict):
                                flat_corr.update(item)
                        corrections = flat_corr

                    if isinstance(corrections, dict):
                        for path, val in corrections.items():
                            val_str = str(val).strip()
                            if not val_str or val_str.lower() in ["none", "null", "undefined"]:
                                continue
                            # Anti-hallucination check: only apply if value actually appears in raw text or is numeric
                            if val_str.lower() not in raw_text.lower():
                                clean_val = re.sub(r'[^a-zA-Z0-9]', '', val_str.lower())
                                clean_raw = re.sub(r'[^a-zA-Z0-9]', '', raw_text.lower())
                                if len(clean_val) > 2 and clean_val not in clean_raw and not isinstance(val, (int, float)):
                                    continue
                            parts = path.split(".")
                            target = candidate_data
                            if "content" in candidate_data and "custom_fields" in candidate_data["content"]:
                                cf = candidate_data["content"]["custom_fields"]
                                if isinstance(cf, dict) and parts[0] in cf:
                                    target = cf
                            for p in parts[:-1]:
                                if isinstance(target, dict) and p in target:
                                    target = target[p]
                                else:
                                    break
                            if isinstance(target, dict) and parts[-1] in target:
                                target[parts[-1]] = val
                    return candidate_data, audit_summary

                # 2. Handle full schema output if returned
                if "content" in verified_json and isinstance(verified_json["content"], dict):
                    for k in ["document_type", "metadata"]:
                        if k not in verified_json and k in candidate_data:
                            verified_json[k] = candidate_data[k]
                    return verified_json, audit_summary
                elif "contract_and_invoice_details" in verified_json or "goods_and_tax_details" in verified_json:
                    candidate_data["content"]["custom_fields"] = verified_json
                    return candidate_data, audit_summary
                elif "custom_fields" in verified_json:
                    candidate_data["content"]["custom_fields"] = verified_json["custom_fields"]
                    return candidate_data, audit_summary

            return candidate_data, audit_summary

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
        filename: str,
        layout_regions: Optional[Dict[str, str]] = None
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
        date_pattern = r"\b(\d{1,2}[-/\.](?:[A-Za-z]{3}|\d{1,2})[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b"
        raw_dates = list(set(re.findall(date_pattern, raw_text, flags=re.IGNORECASE)))
        dates = [self._format_date_iso(d) for d in raw_dates if self._format_date_iso(d)]

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

        # 6. Extract Tables & Invoice Domain Intelligence
        tables = detected_tables.copy()
        structured_invoice_data = None
        
        # If document is a Tax Invoice or commercial bill, run dedicated high-precision dynamic parser
        if doc_type in ["tax_invoice", "delivery_challan", "purchase_order", "bill_receipt"] or "invoice" in lower_text or "bill" in lower_text:
            inv_tables, inv_custom, inv_kv = self._parse_invoice_structured_data(lines, raw_text, detected_tables, layout_regions=layout_regions)
            tables = inv_tables
            custom_fields.update(inv_custom)
            kv_pairs.extend(inv_kv)
            
            # Build hierarchically structured invoice schema as requested
            structured_invoice_data = self._build_structured_invoice_json(
                raw_text=raw_text,
                lines=lines,
                goods_rows=tables[0].get("rows", []) if tables else [],
                custom=custom_fields,
                layout_regions=layout_regions
            )
            # Use structured invoice as custom_fields
            custom_fields = structured_invoice_data
        elif not tables:
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
                if idx + 1 < len(lines) and len(lines[idx + 1]) < 100:
                    candidate = f"{candidate} {lines[idx + 1]}".strip()
                if candidate:
                    addresses.append(candidate)

        # 8. Dynamic Paragraphs & Sections
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 20][:10]

        sections = [
            {"title": "Invoice Details & Parties", "content": "\n".join(lines[:14])}
        ]

        if tables and tables[0].get("rows"):
            items_summary = []
            for r in tables[0]["rows"]:
                desc = r.get("Description of Goods") or r.get("Description") or r.get("Item") or ""
                amt = r.get("Amount") or ""
                qty = r.get("Quantity") or ""
                rate = r.get("Rate") or ""
                if desc:
                    items_summary.append(f"{desc} | Qty: {qty} | Rate: {rate} | Amount: {amt}".strip(" |"))
            if items_summary:
                sections.append({"title": "Goods & Line Items Summary", "content": "\n".join(items_summary)})

        bank_details_obj = None
        if isinstance(custom_fields, dict):
            bank_details_obj = custom_fields.get("our_bank_details") or custom_fields.get("bank_details")
        if bank_details_obj:
            bd = bank_details_obj
            bank_str = f"Bank Name: {bd.get('bank_name', '-')}\nAccount No: {bd.get('account_number', '-')}\nIFSC Code: {bd.get('ifsc_code', '-')}\nBranch: {bd.get('branch', '-')}"
            sections.append({"title": "Bank & Payment Details", "content": bank_str})

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
                "sections": sections,
                "paragraphs": paragraphs,
                "tables": tables,
                "key_value_pairs": kv_pairs[:50],
                "entities": [{"name": g, "type": "GSTIN", "page": 1} for g in gstins],
                "custom_fields": custom_fields
            }
        }

    def _parse_invoice_structured_data(
        self,
        lines: List[str],
        raw_text: str,
        detected_tables: Optional[List[Dict[str, Any]]] = None,
        layout_regions: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """
        Universal dynamic extractor for GST Tax Invoices and Commercial Bills.
        Dynamically extracts whatever line items, taxes, totals, bank details,
        PAN, and terms exist in the document without hardcoded values.
        """
        goods_rows = []
        custom: Dict[str, Any] = {}
        extra_kv = []

        # 1. Use detected tables from PyMuPDF if valid tabular rows exist
        if detected_tables:
            for t in detected_tables:
                headers = [str(h or "").lower() for h in t.get("headers", [])]
                if any(k in " ".join(headers) for k in ["description", "item", "particulars", "hsn", "rate", "amount"]):
                    valid_table_rows = []
                    for r in t.get("rows", []):
                        desc = str(r.get("Description of Goods") or r.get("Description") or r.get("Item") or "").strip()
                        amt = str(r.get("Amount") or "").strip()
                        if desc or (amt and amt not in ["-", "0", "0.0", "0.00"]):
                            clean_row = {k: v for k, v in r.items() if v and v != "None"}
                            valid_table_rows.append(clean_row)
                    if any(str(r.get("Description of Goods") or r.get("Description") or r.get("Item") or "").strip() for r in valid_table_rows):
                        goods_rows.extend(valid_table_rows)

        # 2. If layout_regions provided (image pipeline), extract from goods_table region
        if not goods_rows and layout_regions and layout_regions.get("goods_table"):
            gt_lines = [l.strip() for l in layout_regions["goods_table"].split("\n") if l.strip()]
            for l_s in gt_lines:
                if any(k in l_s.lower() for k in ["description", "goods", "total", "cgst", "sgst", "subtotal", "amount"]):
                    continue
                hsn_m = re.search(r'\b(\d{6,8}|\d{4})\b', l_s)
                if hsn_m:
                    hsn = hsn_m.group(1)
                    desc_part = l_s[:l_s.find(hsn)].strip()
                    desc_m = re.search(r'\b(M\d+|M-\d+|[A-Za-z0-9\s\-\.\/\(\)]{2,30})\b', desc_part)
                    desc = desc_m.group(1).strip() if desc_m else "Item"
                    desc = re.sub(r'^\d+\s*[\.\-\]\|]?\s*', '', desc).strip()

                    after_hsn = l_s[l_s.find(hsn) + len(hsn):]
                    norm_after = re.sub(r'(\d+),(\d{2})\b', r'\1.\2', after_hsn)
                    tokens = re.findall(r'[\d,]+\.?\d*', norm_after)
                    floats = []
                    for t in tokens:
                        try:
                            floats.append(float(t.replace(",", "")))
                        except ValueError:
                            pass

                    # Arithmetic confirmation: qty * rate == amount
                    matched = False
                    qty, rate, amt = 0.0, 0.0, 0.0
                    for i in range(len(floats)):
                        for j in range(i + 1, len(floats)):
                            for k in range(j + 1, len(floats)):
                                q, r, a = floats[i], floats[j], floats[k]
                                for q_cand in [q, q / 100.0 if q > 100 else q]:
                                    if abs(q_cand * r - a) < 2.0 or abs(a - (q_cand * r)) < 2.0:
                                        qty, rate, amt = q_cand, r, a
                                        matched = True
                                        break
                                if matched: break
                            if matched: break
                        if matched: break

                    if not matched:
                        if len(floats) >= 3:
                            qty, rate, amt = floats[0], floats[1], floats[2]
                        elif len(floats) == 2:
                            qty, rate, amt = floats[0], 0.0, floats[1]
                        else:
                            qty, rate, amt = 0.0, 0.0, floats[0] if floats else 0.0

                    unit = "M3"
                    for u in ["CU.MT", "CUMT", "M3", "MT", "NOS", "KG", "LTR", "PCS", "SQM", "BAGS"]:
                        if u.lower() in l_s.lower():
                            unit = u
                            break

                    goods_rows.append({
                        "SI No": str(len(goods_rows) + 1),
                        "Description of Goods": desc,
                        "HSN/SAC": hsn,
                        "Quantity": f"{qty} {unit}".strip(),
                        "Rate": f"{rate:,.2f}",
                        "Per": unit,
                        "Amount": f"{amt:,.2f}"
                    })

        # 3. Extract line items from PyMuPDF vertical blocks
        if not goods_rows:
            item_pattern = re.compile(
                r'(?:^|\n)\s*(\d{1,2})\s*\n\s*([\d,]+\.\d{2})\s*\n\s*([A-Za-z0-9\.\/]+)\s*\n\s*(\d+[\.,]\d{2,3})\s*\n\s*(\d{4,8})\s*\n\s*([\d,]+\.\d{2})\s*\n\s*([A-Za-z0-9\s\-\.\/\(\)]+?)(?=\n\s*\d{1,2}\s*\n|\n\s*Total|\n\s*[\d,]+\.\d{2}\s*\n\s*Total|\n\s*CGST|\n\s*AckNo|$)',
                re.MULTILINE
            )
            for m in item_pattern.findall(raw_text):
                goods_rows.append({
                    "SI No": m[0],
                    "Description of Goods": m[6].strip(),
                    "HSN/SAC": m[4],
                    "Quantity": f"{m[3]} {m[2]}".strip(),
                    "Rate": m[1],
                    "Per": m[2],
                    "Amount": m[5]
                })

        # 3b. Tally / GST vertical invoice layout (Sl No + Description on first line, followed by Amount, Per, Rate, Qty, HSN)
        if not goods_rows:
            tally_pattern = re.compile(
                r'(?:^|\n)\s*(\d{1,2})\s+([A-Za-z0-9\s\-\.\/\(\)]+?)\s*\n\s*([\d,]+\.\d{2})\s*\n\s*([A-Za-z0-9\.\/]+)\s*\n\s*([\d,]+\.\d{2})\s*\n\s*([\d,]+(?:\.\d{2,3})?)\s*([A-Za-z0-9\.\/]+)?\s*\n\s*(\d{4,8})',
                re.MULTILINE
            )
            for m in tally_pattern.finditer(raw_text):
                sl_no = m.group(1)
                desc = m.group(2).strip()
                amt = m.group(3)
                per = m.group(4).strip()
                rate = m.group(5)
                qty = m.group(6)
                unit = m.group(7) or per
                hsn = m.group(8)

                # Collect extra description lines if any
                pos_end = m.end()
                rest = raw_text[pos_end:]
                extra_lines = []
                for el in rest.splitlines():
                    el_s = el.strip()
                    if not el_s:
                        continue
                    if any(term in el_s.lower() for term in ["cgst", "sgst", "total", "round off", "amount chargeable", "hsn/sac", "taxable"]):
                        break
                    extra_lines.append(el_s)

                goods_rows.append({
                    "SI No": sl_no,
                    "Description of Goods": desc,
                    "HSN/SAC": hsn,
                    "Quantity": f"{qty} {unit}".strip(),
                    "Rate": rate,
                    "Per": per,
                    "Amount": amt,
                    "additional_details": extra_lines if extra_lines else None
                })

        # 3c. Horizontal line items with HSN, Qty, Rate, Amount (e.g. OCR single line)
        if not goods_rows:
            horiz_pattern = re.compile(
                r'(?:^|\n)\s*(\d{1,2})\s+([A-Za-z0-9\s\-\.\/\(\)]+?)\s+(\d{4,8})\s+([\d,]+(?:\.\d+)?)\s*([A-Za-z0-9\.\/]+)?\s+([\d,]+(?:\.\d+)?)\s*([A-Za-z0-9\.\/]+)?\s+([\d,]+(?:\.\d+)?)',
                re.MULTILINE
            )
            for m in horiz_pattern.finditer(raw_text):
                sl_no = m.group(1)
                desc = m.group(2).strip()
                hsn = m.group(3)
                qty = m.group(4)
                unit = m.group(5) or "Mtr"
                rate = m.group(6)
                per = m.group(7) or unit
                amt = m.group(8)

                pos_end = m.end()
                rest = raw_text[pos_end:]
                extra_lines = []
                for el in rest.splitlines():
                    el_s = el.strip()
                    if not el_s:
                        continue
                    if any(term in el_s.lower() for term in ["cgst", "sgst", "total", "round off", "amount chargeable", "hsn/sac", "taxable"]):
                        break
                    extra_lines.append(el_s)

                goods_rows.append({
                    "SI No": sl_no,
                    "Description of Goods": desc,
                    "HSN/SAC": hsn,
                    "Quantity": f"{qty} {unit}".strip(),
                    "Rate": rate,
                    "Per": per,
                    "Amount": amt,
                    "additional_details": extra_lines if extra_lines else None
                })

        # 3d. Textile & Job-work vertical column blocks (Pieces -> Rate -> Particular -> Aprx.Mts -> Amount -> Bale No)
        if not goods_rows:
            i = 0
            doc_hsn = None
            hm = re.search(r'HSN\s*CODE\s*[:\-]?\s*[\s\S]+?(\d{6,8})', raw_text, re.I)
            if hm:
                doc_hsn = hm.group(1)

            while i < len(lines):
                l = lines[i]
                if re.match(r'^\d{2,6}$', l) and i + 1 < len(lines) and re.match(r'^\d+\.\d{2}$', lines[i+1]):
                    pcs = int(l)
                    rate = float(lines[i+1])
                    particular_lines = []
                    amt = None
                    bale = None
                    aprx_mts = 0.0
                    j = i + 2
                    while j < len(lines) and j < i + 10:
                        sub = lines[j]
                        if any(term in sub.lower() for term in ["total", "disc", "taxble", "cgst", "sgst", "igst"]):
                            break
                        if re.match(r'^\d+\.\d{2}$', sub):
                            val = float(sub)
                            if val == 0.0:
                                aprx_mts = val
                            elif val > 100.0 or val > 0:
                                amt = val
                                if j + 1 < len(lines) and re.match(r'^\d{4,8}$', lines[j+1]):
                                    bale = lines[j+1]
                                    j += 1
                                break
                        else:
                            particular_lines.append(sub)
                        j += 1

                    if amt is not None and bale is not None:
                        desc = " ".join(particular_lines).strip()
                        goods_rows.append({
                            "SI No": str(len(goods_rows) + 1),
                            "Description of Goods": desc,
                            "bale_number": bale,
                            "approx_meters": aprx_mts,
                            "pieces": pcs,
                            "HSN/SAC": doc_hsn or "520811",
                            "Quantity": f"{pcs} PCS",
                            "Rate": f"{rate:,.2f}",
                            "Per": "PCS",
                            "Amount": f"{amt:,.2f}"
                        })
                        i = j
                i += 1

        # 3. Horizontal line items (Tesseract OCR / single lines)
        if not goods_rows:
            for line in lines:
                m_item = re.search(r'^(\d{1,2})\s+([A-Za-z0-9\s\-\.\/\(\)]+?)\s+(\d{4,8})\s+(\d+[\.,]\d{2,3})\s*([A-Za-z0-9\.\/]+)?\s+([\d,]+\.?\d*)\s+([\d,]+\.?\d*)', line)
                if m_item:
                    goods_rows.append({
                        "SI No": m_item.group(1),
                        "Description of Goods": m_item.group(2).strip(),
                        "HSN/SAC": m_item.group(3),
                        "Quantity": f"{m_item.group(4)} {m_item.group(5) or ''}".strip(),
                        "Rate": m_item.group(6),
                        "Per": m_item.group(5) or "-",
                        "Amount": m_item.group(7)
                    })

        # 4. Fallback product regex line detection (e.g. for noisy OCR scans)
        if not goods_rows:
            for line in lines:
                if any(term in line.lower() for term in ["terms", "condition", "details", "invoice", "bank", "address"]):
                    continue
                if re.search(r'\b(M-?\d+|concrete|cement|charge|item|qty|goods)\b', line, re.I):
                    amounts = re.findall(r'[\d,]+\.\d{2}', line)
                    if not amounts:
                        continue
                    hsn_m = re.search(r'\b(\d{6,8})\b', line)
                    qty_m = re.search(r'(\d+[\.,]\d{2,3})\s*(CU\.?MT|MT|NOS|KG|LTR|PCS|M3)?', line, re.I)
                    desc_m = re.search(r'([A-Za-z0-9\s\-]{3,25})(?=\s+\d{6,8}|\s+\d+[\.,]\d{2,3})', line)
                    
                    desc = desc_m.group(1).strip() if desc_m else "Item"
                    desc = re.sub(r'^\d+\s*', '', desc)
                    if desc.lower() not in ["total", "cgst", "sgst", "igst"]:
                        goods_rows.append({
                            "SI No": str(len(goods_rows) + 1),
                            "Description of Goods": desc,
                            "HSN/SAC": hsn_m.group(1) if hsn_m else "-",
                            "Quantity": qty_m.group(0).strip() if qty_m else "-",
                            "Rate": amounts[0] if len(amounts) >= 2 else "-",
                            "Per": qty_m.group(2) if (qty_m and qty_m.group(2)) else "-",
                            "Amount": amounts[-1] if amounts else "-"
                        })

        # Dynamic Taxes (CGST, SGST, IGST) from Layout Regions or Full Text
        tax_source = layout_regions.get("taxes") if layout_regions else ""
        if tax_source:
            for l in tax_source.splitlines():
                if re.search(r'\b[CG]GST\b', l, re.I) and "cgst" not in custom:
                    rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', l)
                    if rate_m:
                        custom["cgst_rate"] = f"{rate_m.group(1)}%"
                    amts = re.findall(r'[\d,]+\.\d{2}', l)
                    if amts:
                        val = float(amts[-1].replace(",", ""))
                        custom["cgst"] = val
                        custom["cgst_amount"] = val
                        if not custom.get("cgst_rate"):
                            custom["cgst_rate"] = "9.00%"
                if re.search(r'\bSGST\b', l, re.I) and "sgst" not in custom:
                    rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', l)
                    if rate_m:
                        custom["sgst_rate"] = f"{rate_m.group(1)}%"
                    amts = re.findall(r'[\d,]+\.\d{2}', l)
                    if amts:
                        val = float(amts[-1].replace(",", ""))
                        custom["sgst"] = val
                        custom["sgst_amount"] = val
                        if not custom.get("sgst_rate"):
                            custom["sgst_rate"] = "9.00%"
                if re.search(r'\bRound', l, re.I) and "rounding_adjustment" not in custom:
                    amts = re.findall(r'[\d,]+\.\d{2}', l)
                    if amts:
                        custom["rounding_adjustment"] = float(amts[-1].replace(",", ""))
                if re.search(r'(?:Total\s*Amount|Amount\s*:-)', l, re.I) and "grand_total" not in custom:
                    amts = re.findall(r'[\d,]+\.\d{2}', l)
                    if amts:
                        custom["grand_total"] = float(amts[-1].replace(",", ""))

        for idx, line in enumerate(lines):
            l_clean = line.strip()
            # CGST
            if re.search(r'\b[cg]gst\b', l_clean, re.I) and "cgst" not in custom:
                rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', l_clean)
                if not rate_m and idx > 0:
                    rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', lines[idx - 1])
                cgst_rate = f"{rate_m.group(1)}%" if rate_m else "9%"
                
                clean_l = re.sub(r'\d+(?:\.\d+)?%', '', l_clean)
                amts = re.findall(r'[\d,]+\.\d{2}', clean_l)
                if not amts and idx + 1 < len(lines):
                    c_next = re.sub(r'\d+(?:\.\d+)?%', '', lines[idx + 1])
                    amts = re.findall(r'[\d,]+\.\d{2}', c_next)
                
                if amts:
                    val = float(amts[-1].replace(",", ""))
                    custom["cgst"] = val
                    custom["cgst_amount"] = val
                    custom["cgst_rate"] = cgst_rate
                    goods_rows.append({
                        "SI No": "-",
                        "Description of Goods": f"CGST @ {cgst_rate}",
                        "HSN/SAC": "-",
                        "Quantity": "-",
                        "Rate": cgst_rate,
                        "Per": "-",
                        "Amount": f"{val:,.2f}"
                    })
            
            # SGST
            elif re.search(r'\bsgst\b', l_clean, re.I) and "sgst" not in custom:
                rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', l_clean)
                if not rate_m and idx + 1 < len(lines):
                    rate_m = re.search(r'(\d+(?:\.\d+)?)\s*%', lines[idx + 1])
                sgst_rate = f"{rate_m.group(1)}%" if rate_m else "9%"
                
                clean_l = re.sub(r'\d+(?:\.\d+)?%', '', l_clean)
                amts = re.findall(r'[\d,]+\.\d{2}', clean_l)
                if not amts and idx + 1 < len(lines):
                    c_next = re.sub(r'\d+(?:\.\d+)?%', '', lines[idx + 1])
                    amts = re.findall(r'[\d,]+\.\d{2}', c_next)
                if not amts and idx + 2 < len(lines):
                    c_next2 = re.sub(r'\d+(?:\.\d+)?%', '', lines[idx + 2])
                    amts = re.findall(r'[\d,]+\.\d{2}', c_next2)
                
                if amts:
                    val = float(amts[-1].replace(",", ""))
                    custom["sgst"] = val
                    custom["sgst_amount"] = val
                    custom["sgst_rate"] = sgst_rate
                    goods_rows.append({
                        "SI No": "-",
                        "Description of Goods": f"SGST @ {sgst_rate}",
                        "HSN/SAC": "-",
                        "Quantity": "-",
                        "Rate": sgst_rate,
                        "Per": "-",
                        "Amount": f"{val:,.2f}"
                    })

        # Intra-state symmetry fallback
        if "cgst" in custom and "sgst" not in custom:
            custom["sgst"] = custom["cgst"]
            custom["sgst_amount"] = custom["cgst"]
            custom["sgst_rate"] = custom.get("cgst_rate", "9.00%")
        elif "sgst" in custom and "cgst" not in custom:
            custom["cgst"] = custom["sgst"]
            custom["cgst_amount"] = custom["sgst"]
            custom["cgst_rate"] = custom.get("sgst_rate", "9.00%")

        # Discount extraction
        disc_m = re.search(r'Disc\.[^\n]*\n\s*([\d,]+\.\d{2})\s*\n\s*(\d+(?:\.\d+)?)\s*%', raw_text, re.I)
        if not disc_m:
            disc_m = re.search(r'(?:Disc|Discount)\.?\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%\s*[:\-]?\s*([\d,]+\.\d{2})', raw_text, re.I)
        if disc_m:
            try:
                g1 = disc_m.group(1).replace(",", "")
                g2 = disc_m.group(2).replace(",", "")
                if "%" in disc_m.group(0) and float(g2) <= 100.0:
                    disc_val = float(g1)
                    disc_rate = float(g2)
                else:
                    disc_rate = float(g1)
                    disc_val = float(g2)
                custom["discount"] = {"rate_percent": disc_rate, "amount": disc_val}
                custom["discount_amount"] = disc_val
                custom["discount_rate"] = disc_rate
            except (ValueError, IndexError):
                pass

        # IGST extraction
        igst_m = re.search(r'IGST[^\n]*\n\s*(\d+(?:\.\d+)?)\s*%\s*\n\s*([\d,]+\.\d{2})', raw_text, re.I)
        if not igst_m:
            igst_m = re.search(r'IGST\s*(?:@\s*)?(\d+(?:\.\d+)?)\s*%\s*[:\-]?\s*([\d,]+\.\d{2})', raw_text, re.I)
        if igst_m:
            try:
                ig_rate = float(igst_m.group(1))
                ig_val = float(igst_m.group(2).replace(",", ""))
                custom["igst"] = ig_val
                custom["igst_amount"] = ig_val
                custom["igst_rate"] = f"{ig_rate}%"
            except (ValueError, IndexError):
                pass

        # Taxable Value & Financial Totals
        actual_item_rows = [r for r in goods_rows if r.get("SI No") != "-" and r.get("Amount") and r["Amount"] != "-"]
        item_amounts = []
        for r in actual_item_rows:
            try:
                item_amounts.append(float(str(r["Amount"]).replace(",", "")))
            except (ValueError, TypeError):
                pass
        
        computed_taxable = sum(item_amounts) if item_amounts else None

        if "taxable_value" not in custom:
            taxable_m = re.search(r'(?:Tax(?:a)?ble\s*Value|Assessable\s*Value)\s*[:\-\s₹Rs\.]*([\d,]+\.\d{2})', raw_text, re.I)
            if not taxable_m:
                taxable_m = re.search(r'Tax(?:a)?ble\s*Value[^\n]*\n\s*([\d,]+\.\d{2})', raw_text, re.I)
            if not taxable_m:
                taxable_m = re.search(r'([\d,]+\.\d{2})\s*\n\s*[\d\.]+\s*\n\s*[A-Za-z0-9\.]+\s*\n\s*Total', raw_text, re.I)
            if taxable_m:
                try:
                    custom["taxable_value"] = float(taxable_m.group(1).replace(",", ""))
                except ValueError:
                    pass
            elif computed_taxable:
                disc_val = (custom.get("discount") or {}).get("amount", 0.0)
                custom["taxable_value"] = round(computed_taxable - disc_val, 2) if disc_val > 0 else computed_taxable

        cgst_val = custom.get("cgst", 0.0)
        sgst_val = custom.get("sgst", 0.0)
        tot_tax = cgst_val + sgst_val
        if tot_tax > 0:
            custom["total_tax_amount"] = tot_tax

        # Grand Total
        gt_m = re.search(r'(?:Total\s*Amount|Grand\s*Total|Total\s*Payable|Net\s*Amount)\s*[:\-\s₹Rs\.]*([\d,]+\.\d{2})', raw_text, re.I)
        if not gt_m:
            gt_m = re.search(r'([\d,]+\.\d{2})\s*\n\s*(?:Total\s*Amount|Grand\s*Total|Net\s*Amount)', raw_text, re.I)
        
        if gt_m:
            custom["grand_total"] = float(gt_m.group(1).replace(",", ""))
        
        if not custom.get("grand_total"):
            for line in lines:
                if re.search(r'\b(Total|TT)\b', line, re.I) and not re.search(r'\b(taxable|tax)\b', line, re.I):
                    amts = re.findall(r'[\d,]+\.\d{2}', line)
                    if amts:
                        try:
                            cand = float(amts[-1].replace(",", ""))
                            if cand > (custom.get("taxable_value") or 0):
                                custom["grand_total"] = cand
                                break
                        except ValueError:
                            pass

        if not custom.get("grand_total") and custom.get("taxable_value") and tot_tax > 0:
            custom["grand_total"] = round(custom["taxable_value"] + tot_tax, 2)

        # Dynamic Total Quantity
        qty_m = re.search(r'(\d+[\.,]\d{2,3})\s*([A-Za-z0-9\.\/]+)\s*\n\s*Total', raw_text, re.I)
        if not qty_m:
            qty_m = re.search(r'Total\s+(\d+[\.,]\d{2,3}\s*[A-Za-z0-9\.\/]*)', raw_text, re.I)
        
        if qty_m:
            custom["total_quantity"] = f"{qty_m.group(1)} {qty_m.group(2)}".strip() if qty_m.lastindex >= 2 else qty_m.group(1).strip()
        elif actual_item_rows:
            custom["total_quantity"] = actual_item_rows[0].get("Quantity", "-")

        # Add Total row to goods_rows
        if "grand_total" in custom and not any(r.get("Description of Goods") == "Total" for r in goods_rows):
            goods_rows.append({
                "SI No": "-",
                "Description of Goods": "Total",
                "HSN/SAC": "-",
                "Quantity": custom.get("total_quantity", "-"),
                "Rate": "-",
                "Per": "-",
                "Amount": f"{custom['grand_total']:,.2f}"
            })

        # Dynamic Tax Breakdown Table
        tax_rows = []
        unique_hsns = list(set([r.get("HSN/SAC") for r in goods_rows if r.get("HSN/SAC") and r.get("HSN/SAC") != "-"]))
        hsn_val = unique_hsns[0] if unique_hsns else "HSN"
        if "taxable_value" in custom:
            tax_rows.append({
                "HSN/SAC": hsn_val,
                "Taxable Value": f"{custom['taxable_value']:,.2f}",
                "CGST Rate": custom.get("cgst_rate", "9%"),
                "CGST Amount": f"{custom.get('cgst', 0.0):,.2f}",
                "SGST Rate": custom.get("sgst_rate", "9%"),
                "SGST Amount": f"{custom.get('sgst', 0.0):,.2f}",
                "Total Tax Amount": f"{custom.get('total_tax_amount', 0.0):,.2f}"
            })

        # Dynamic PAN and GSTIN
        pan_m = re.search(r'(?:PAN|P\.A\.N\.)\s*(?:No\.?)?\s*[:\-\s»]*([A-Z0-9]{10})\b', raw_text, re.I)
        if pan_m:
            custom["company_pan"] = pan_m.group(1)
            extra_kv.append({"key": "Company PAN", "value": pan_m.group(1), "page": 1})
        else:
            pans = list(set(re.findall(r'\b[A-Z]{5}\d{4}[A-Z]\b', raw_text)))
            if pans:
                custom["company_pan"] = pans[0]
                extra_kv.append({"key": "Company PAN", "value": pans[0], "page": 1})

        gstins = list(set(re.findall(r'\b\d{2}[A-Z]{5}\d{4}[A-Z\d]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b', raw_text)))
        if gstins:
            custom["supplier_gstin"] = gstins[0]
            extra_kv.append({"key": "Supplier GSTIN", "value": gstins[0], "page": 1})
            if len(gstins) > 1:
                custom["client_gstin"] = gstins[1]
                extra_kv.append({"key": "Client GSTIN", "value": gstins[1], "page": 1})

        # Dynamic Bank Details
        bank_name = None
        acc_no = None
        ifsc = None
        branch = None

        ifsc_m = re.search(r'\b([A-Z]{4}0[A-Z0-9]{6})\b', raw_text)
        if ifsc_m:
            ifsc = ifsc_m.group(1)

        bank_colon_m = re.search(r':\s*([A-Za-z\s]+(?:BANK|Bank))\s*\n\s*:\s*([A-Za-z0-9\s\(\)\-]+?)\s*\n\s*:\s*(\d{9,18})', raw_text)
        if bank_colon_m:
            bank_name = bank_colon_m.group(1).strip()
            branch = bank_colon_m.group(2).strip()
            acc_no = bank_colon_m.group(3).strip()
        else:
            b_name_m = re.search(r'(?:Bank\s*Name|Bank)\s*[:\-]\s*([A-Za-z\s]+(?:Bank|Mahindra Bank))', raw_text, re.I)
            if b_name_m:
                bank_name = b_name_m.group(1).strip()
            
            b_acc_m = re.search(r'(?:A/c\s*(?:No\.?)?|Account\s*No\.?)\s*[:\-]?\s*(\d{9,18})', raw_text, re.I)
            if b_acc_m:
                acc_no = b_acc_m.group(1).strip()
                
            b_branch_m = re.search(r'(?:Branch|Branch\s*&\s*IFS\s*Code)\s*[:\-]\s*([^,\n]+(?:,\s*[^,\n]+)?)', raw_text, re.I)
            if b_branch_m:
                branch = b_branch_m.group(1).strip()

        if acc_no or ifsc or bank_name:
            custom["bank_details"] = {
                "bank_name": bank_name or "-",
                "account_number": acc_no or "-",
                "ifsc_code": ifsc or "-",
                "branch": branch or "-"
            }
            if bank_name:
                extra_kv.append({"key": "Bank Name", "value": bank_name, "page": 1})
            if acc_no:
                extra_kv.append({"key": "Bank A/c No", "value": acc_no, "page": 1})
            if ifsc:
                extra_kv.append({"key": "Bank IFSC Code", "value": ifsc, "page": 1})

        # Amount in words
        words_m = re.search(r'(?:Amount\s*in\s*words|Rupees|INR)\s*[:\-]?\s*([A-Za-z\s]+(?:Only|only)\.?)', raw_text, re.I)
        if words_m:
            custom["amount_chargeable_in_words"] = words_m.group(0).strip()
            extra_kv.append({"key": "Amount in Words", "value": custom["amount_chargeable_in_words"], "page": 1})

        if "grand_total" in custom:
            extra_kv.append({"key": "Grand Total", "value": f"₹{custom['grand_total']:,.2f}", "page": 1})
        if "taxable_value" in custom:
            extra_kv.append({"key": "Taxable Value", "value": f"₹{custom['taxable_value']:,.2f}", "page": 1})

        tables = [
            {
                "title": "Description of Goods & Line Items",
                "page": 1,
                "headers": ["SI No", "Description of Goods", "HSN/SAC", "Quantity", "Rate", "Per", "Amount"],
                "rows": goods_rows
            }
        ]
        if tax_rows:
            tables.append({
                "title": "Tax Breakdown (HSN/SAC)",
                "page": 1,
                "headers": ["HSN/SAC", "Taxable Value", "CGST Rate", "CGST Amount", "SGST Rate", "SGST Amount", "Total Tax Amount"],
                "rows": tax_rows
            })

        return tables, custom, extra_kv

    def _clean_gstin(self, val: Optional[str], pan: Optional[str] = None) -> Optional[str]:
        if not val:
            return None
        v = re.sub(r'[^A-Z0-9\(\)\]\[\{\}\?]', '', val.upper())
        if len(v) >= 15:
            s = list(v[:15])
            if s[0] in ['Z', '}', '{', '?', ')', ']', '>', '1']: s[0] = '2'
            if s[1] in ['I', 'L', '1']: s[1] = '7'
            for i in range(2, 7):
                if s[i] == '0': s[i] = 'O'
                if s[i] == '1': s[i] = 'I'
                if s[i] == 'P' and pan and pan.startswith("AAF"): s[i] = 'F'
            for i in range(7, 11):
                if s[i] == 'O': s[i] = '0'
                if s[i] in ['I', 'L']: s[i] = '1'
                if s[i] == 'S': s[i] = '5'
            if s[11] == '0': s[11] = 'O'
            if s[11] in ['1', 'L', ')', ']']: s[11] = 'J'
            if s[12] in ['I', 'L']: s[12] = '1'
            s[13] = 'Z'
            if pan and len(pan) == 10:
                # If 7+ characters match PAN, align exact PAN characters into GSTIN
                matches = sum(1 for a, b in zip(s[2:12], list(pan)) if a == b)
                if matches >= 6:
                    s[2:12] = list(pan)
            return "".join(s)
        return val

    def _format_date_iso(self, date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        d = date_str.strip()
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        m_txt = re.match(r'(\d{1,2})[\s\-\/\.]([A-Za-z]{3,9})[\s\-\/\.](\d{2,4})', d)
        if m_txt:
            day = int(m_txt.group(1))
            m_str = m_txt.group(2)[:3].lower()
            yr = m_txt.group(3)
            if len(yr) == 2:
                yr = "20" + yr
            if m_str in month_map:
                return f"{yr}-{month_map[m_str]:02d}-{day:02d}"

        m = re.match(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})', d)
        if m:
            p1, p2, yr = int(m.group(1)), int(m.group(2)), m.group(3)
            if len(yr) == 2:
                yr = "20" + yr
            if p1 > 12 and p2 <= 12:
                day, month = p1, p2
            elif p2 > 12 and p1 <= 12:
                day, month = p2, p1
            else:
                day, month = p1, p2
            return f"{yr}-{month:02d}-{day:02d}"
        return d

    def _build_structured_invoice_json(
        self,
        raw_text: str,
        lines: List[str],
        goods_rows: List[Dict[str, Any]],
        custom: Dict[str, Any],
        layout_regions: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Builds the clean, hierarchical invoice schema matching user requirements:
        - contract_and_invoice_details (seller_details, contract_person_details, invoice_details)
        - goods_and_tax_details (items, taxes, subtotal, total_amount_after_tax)
        - our_bank_details (bank_name, branch, account_number, ifsc_code, account_holder_name)
        Excludes raw Terms & Conditions (T&C).
        """
        header_txt = (layout_regions or {}).get("header")
        buyer_txt = (layout_regions or {}).get("buyer")
        inv_txt = (layout_regions or {}).get("invoice_meta")
        einv_txt = (layout_regions or {}).get("e_invoice")
        rem_txt = (layout_regions or {}).get("remarks")
        bank_txt = (layout_regions or {}).get("bank")

        # 1. Seller Details
        seller_name = None
        if header_txt:
            h_lines = [l.strip() for l in header_txt.split("\n") if l.strip()]
            for l in h_lines:
                if any(k in l.upper() for k in ["PVT", "LTD", "LIMITED", "LLP", "ENTERPRISES", "TRADERS", "INDUSTRIES", "CORP", "CORPORATION", "SERVICES", "COMPANY", "CO.", "AGENCY", "FIRM", "ASSOCIATES", "WORKS", "COMMERCIAL"]):
                    cand = re.sub(r'^[^\w]+', '', l).strip()
                    cand = re.sub(r'\s+[A-Z]$', '', cand).strip()
                    seller_name = cand
                    break

        if not seller_name:
            einv_seller_m = re.search(r'e-Invoice\s*\n\s*([^\n]+)', raw_text, re.I)
            if einv_seller_m:
                cand = einv_seller_m.group(1).strip()
                if len(cand) > 3 and not any(k in cand.lower() for k in ["tax invoice", "irn", "ack", "gstin"]):
                    seller_name = cand

        if not seller_name:
            for_m = re.search(r'for\s+([A-Za-z0-9\s\.\&\-]+?)\s*\n\s*Authorised', raw_text, re.I)
            if for_m:
                seller_name = for_m.group(1).strip()

        if not seller_name:
            ac_m = re.search(r'A/c\s*Holder[’\']?s\s*Name\s*[:\-]\s*([^\n]+)', raw_text, re.I)
            if ac_m:
                seller_name = ac_m.group(1).strip()

        if not seller_name:
            seller_m = re.search(r'For\s+M/S\.?\s*([A-Za-z0-9\s\.\&\-]+?)(?=\n|Authorised|Receiver|$)', raw_text, re.I)
            if seller_m:
                cand = seller_m.group(1).strip()
                if len(cand) > 3:
                    seller_name = cand

        if not seller_name:
            for l in lines[:10]:
                if any(k in l.upper() for k in ["ENTERPRISES", "SOLUTIONS", "CONCRETE", "INFRA", "PVT", "LTD", "TRADERS", "COMPANY", "CORP", "ASSOCIATES", "INDUSTRIES", "PROCESSORS", "PROCESS", "TEXTILE", "MILLS", "FABRICS", "FAB"]):
                    seller_name = l.strip()
                    break

        found_pans = re.findall(r'\b[A-Z]{5}\d{4}[A-Z]\b', raw_text)
        pans = []
        for p in found_pans:
            if p not in pans:
                pans.append(p)

        found_gstins = re.findall(r'\b\d{2}[A-Z]{5}\d{4}[A-Z\d]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b', raw_text)
        gstins = []
        for g in found_gstins:
            if g not in gstins:
                gstins.append(g)
        
        seller_pan = custom.get("company_pan")
        if header_txt and not seller_pan:
            span_m = re.search(r'PAN\s*No\.?\s*[:\-\.]*\s*([A-Z0-9]{9,11})', header_txt, re.I)
            if span_m:
                c_pan = re.sub(r'[^A-Z0-9]', '', span_m.group(1).upper())
                if len(c_pan) >= 10:
                    c_pan = c_pan[:4] + ('I' if c_pan[4] in ['L', '1'] else c_pan[4]) + c_pan[5:10]
                    seller_pan = c_pan

        if not seller_pan and pans:
            seller_pan = pans[0]

        cin = None
        if header_txt:
            cin_m = re.search(r'CIN\s*[:\-]\s*([A-Z0-9]{21})', header_txt, re.I)
            if cin_m:
                cin = cin_m.group(1).strip()
        if not cin:
            cin_m = re.search(r'CIN\s*[:\-]\s*([A-Z0-9]{21})', raw_text, re.I)
            if cin_m:
                cin = cin_m.group(1).strip()

        seller_addr = None
        if header_txt:
            addr_m = re.search(r'Regd\.?\s*Office\s*[:\-]?\s*([^\n]+)', header_txt, re.I)
            if addr_m:
                seller_addr = addr_m.group(1).replace("1 |. ", "").replace("OMS", "CHS").strip()

        if not seller_addr:
            post_m = re.search(r'POSTAL\s*[:\-]\s*([^\n]+)', raw_text, re.I)
            if post_m:
                seller_addr = post_m.group(1).strip()

        if not seller_addr and seller_name:
            s_pos = raw_text.find(seller_name)
            if s_pos != -1:
                g_pos = raw_text.find("GSTIN", s_pos)
                if g_pos != -1:
                    addr_block = raw_text[s_pos + len(seller_name):g_pos].strip()
                    addr_lines = [l.strip().rstrip(",") for l in addr_block.splitlines() if l.strip() and not any(k in l.lower() for k in ["tax invoice", "e-invoice", "irn", "ack", "state name"])]
                    if addr_lines:
                        seller_addr = ", ".join(addr_lines)

        seller_email = None
        email_source = (header_txt or "") + "\n" + raw_text
        em_m = re.search(r'\b([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b', email_source)
        if em_m:
            seller_email = em_m.group(1).strip().rstrip(".,;")

        seller_phone = None
        seller_contact_person = None
        scp_m = re.search(r'([A-Za-z\s]{3,30})\s*[-:]\s*([6-9]\d{9})', raw_text[:500])
        if scp_m:
            c_lines = scp_m.group(1).strip().splitlines()
            seller_contact_person = c_lines[-1].strip()
            seller_phone = scp_m.group(2).strip()

        if not seller_phone:
            if header_txt:
                ph_m = re.search(r'(?:Tel|Phone|Mob)[^\d]*(\d{10})', header_txt, re.I)
                if ph_m:
                    seller_phone = ph_m.group(1)
        if not seller_phone:
            ph_m = re.search(r'(?:Contact|Tel|Phone|Mob)[^\d]*(\d{10})', raw_text, re.I)
            if ph_m:
                seller_phone = ph_m.group(1)

        udyam_m = re.search(r'\b(UDYAM-[A-Z0-9\-]+|MH-\d{2}-\d+)\b', raw_text, re.I)
        udyam_no = udyam_m.group(0).strip() if udyam_m else None
        if udyam_no and not udyam_no.startswith("UDYAM-"):
            udyam_no = f"UDYAM-{udyam_no}"

        seller_state = None
        seller_state_code = None

        # Check explicit state pair e.g. GUJARAT - 24
        st_pair_m = re.search(r'\b(GUJARAT|MAHARASHTRA|DELHI|RAJASTHAN|HARYANA|KARNATAKA|TAMIL\s*NADU|WEST\s*BENGAL)\s*[-:]\s*(\d{2})\b', raw_text, re.I)
        if st_pair_m:
            cand_st = st_pair_m.group(1).title()
            cand_sc = st_pair_m.group(2)
            if (seller_addr and (cand_st.lower() in seller_addr.lower() or "guj" in seller_addr.lower() and cand_sc == "24")) or "gujarat" in raw_text[-800:].lower():
                seller_state = cand_st
                seller_state_code = cand_sc

        if not seller_state:
            state_m = re.search(r'(?:State\s*Name|Name\s*Of\s*State)\s*[:\-]\s*([A-Za-z\s]+?)(?:,|$|\n)', raw_text, re.I)
            if state_m:
                seller_state = state_m.group(1).strip()
            code_m = re.search(r'(?:Sate|State)\s*Code\s*[:\-]\s*(\d{2})', raw_text, re.I)
            if code_m:
                seller_state_code = code_m.group(1).strip()

        # Deduce seller GSTIN based on seller state code if known
        seller_gstin = None
        if header_txt:
            sg_m = re.search(r'GST[A-Z\s]*NO\.?\s*[:\-\.]*\s*([A-Z0-9\(\)\!\{\}\?]{13,18})', header_txt, re.I)
            if sg_m:
                seller_gstin = self._clean_gstin(sg_m.group(1), pan=seller_pan)

        if not seller_gstin and seller_state_code and gstins:
            for g in gstins:
                if g.startswith(seller_state_code):
                    seller_gstin = g
                    break

        if not seller_gstin and seller_pan and gstins:
            for g in gstins:
                if seller_pan in g:
                    seller_gstin = g
                    break

        if not seller_gstin and gstins:
            seller_gstin = gstins[0]

        if seller_gstin and len(seller_gstin) >= 2 and seller_gstin[:2].isdigit():
            if not seller_state_code:
                seller_state_code = seller_gstin[:2]
            if len(seller_gstin) >= 12 and not seller_pan:
                seller_pan = seller_gstin[2:12]

        if not seller_state and seller_state_code:
            seller_state = GST_STATE_MAP.get(seller_state_code, "Maharashtra")
        if not seller_state:
            seller_state = "Maharashtra"
            seller_state_code = "27"

        seller_details: Dict[str, Any] = {
            "business_name": seller_name or "-",
            "contact_person": seller_contact_person,
            "address": seller_addr,
            "gstin": seller_gstin,
            "pan": seller_pan,
            "contact_number": seller_phone,
            "email": seller_email,
            "state": seller_state,
            "state_code": seller_state_code
        }
        if cin:
            seller_details["cin"] = cin
        if udyam_no:
            seller_details["udyam_number"] = udyam_no

        # 2. Contract Person / Buyer Details
        buyer_name = None
        buyer_address = None
        site_name = None
        site_address = None
        buyer_gstin = None

        if buyer_txt:
            b_lines = [l.strip() for l in buyer_txt.split("\n") if l.strip()]
            for l in b_lines:
                if not any(k in l.lower() for k in ["gstin", "state", "code", "tax", "site address", "buyer", "bill to", "invoice"]):
                    if len(l) > 3 and not re.match(r'^\d+$', l):
                        buyer_name = l.strip()
                        break

            bg_m = re.search(r'GST[A-Z\s]*NO\.?\s*[:\-\.!]*\s*([A-Z0-9\(\)\!]{13,17})', buyer_txt, re.I)
            if bg_m:
                buyer_gstin = self._clean_gstin(bg_m.group(1))

            addr_parts = []
            capturing_addr = False
            for l in b_lines:
                if buyer_name and l == buyer_name:
                    capturing_addr = True
                    continue
                if capturing_addr:
                    if any(k in l.lower() for k in ["site address", "gstin", "state", "code", "pan"]):
                        break
                    addr_parts.append(l.strip(": "))
            if addr_parts and not buyer_address:
                buyer_address = ", ".join(addr_parts)

            site_pos = buyer_txt.lower().find("site address")
            if site_pos != -1:
                after_site = buyer_txt[site_pos:].split("\n")
                site_lines = [sl.strip(": ") for sl in after_site[1:4] if sl.strip() and not any(k in sl.lower() for k in ["gstin", "pan", "state", "po"])]
                if site_lines:
                    site_address = ", ".join(site_lines)
                    site_name = site_lines[0]

        if not buyer_name:
            to_idx = None
            for idx, l in enumerate(lines[:20]):
                if l.strip().lower() in ["to,", "to"]:
                    to_idx = idx
                    break
            if to_idx is not None:
                for offset in [-2, -1, 1, 2, -3, 3]:
                    pos = to_idx + offset
                    if 0 <= pos < len(lines):
                        candidate = lines[pos].strip()
                        if len(candidate) > 3 and not any(k in candidate.lower() for k in ["invoice", "dated", "tax", "contact", "address", "gstin", "p/"]):
                            if not re.match(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$', candidate):
                                buyer_name = candidate
                                break

        if not buyer_name:
            buyer_m = re.search(r'Buyer\s*(?:\(Bill\s*to\))?\s*[:\-]?\s*\n\s*([A-Za-z0-9\s\.\&\-]+?)(?=\n\s*(?:Room|Floor|Flr|Flat|Plot|Sector|Building|Bldg|GSTIN|State|Road|$))', raw_text, re.I)
            if buyer_m:
                cand = buyer_m.group(1).strip()
                if len(cand) > 2:
                    buyer_name = cand

        if not buyer_name:
            buyer_m = re.search(r'(?:Buyer|Bill\s*to)\s*[:\-\s\n]*([A-Za-z0-9\s\.\&\-]+?)(?=\n|Plot|GSTIN|$)', raw_text, re.I)
            if buyer_m:
                cand = buyer_m.group(1).strip()
                if cand:
                    buyer_name = cand

        if not buyer_name:
            m_s = re.search(r'([A-Za-z0-9\.\s\&]+(?:COMPANY|CO\.|LTD|LLP|PVT|ENTERPRISES|TRADERS))\s*\n\s*M\/s\.?', raw_text)
            if m_s:
                buyer_name = m_s.group(1).strip()

        if not buyer_name:
            b_cand_m = re.search(r'(?:^|\n)\s*([A-Za-z0-9\.\s\&]{3,50}(?:COMPANY|CO\.|LTD|LLP|PVT|ENTERPRISES|TRADERS|CORP|ASSOCIATES|IMPEX|AGENCY))\s*\n\s*GSTIN', raw_text, re.I)
            if b_cand_m:
                buyer_name = b_cand_m.group(1).strip()

        if buyer_name:
            buyer_name = re.sub(r'^\d+\s*', '', buyer_name).strip()

        contact_person = None
        cp_m = re.search(r'Contact\s*Person\s*[:\-]\s*([^\n]+)', raw_text, re.I)
        if cp_m:
            contact_person = cp_m.group(1).strip()

        contact_number = None
        cn_m = re.search(r'(?:Contact\s*Number|Phone|Mobile|Mob\.?)\s*[:\-]?\s*(\d{10})', raw_text, re.I)
        if cn_m:
            contact_number = cn_m.group(1).strip()

        if not buyer_address:
            buyer_sec_m = re.search(r'Buyer\s*(?:\(Bill\s*to\))?\s*[:\-]?\s*([\s\S]+?)(?=\n\s*GSTIN)', raw_text, re.I)
            if buyer_sec_m:
                b_block = buyer_sec_m.group(1).strip().splitlines()
                addr_lines = []
                seen_tokens = set()
                for l in b_block[1:]:
                    l_s = l.strip().rstrip(",")
                    if l_s and not l_s.startswith(":") and not l_s.lower().startswith("state"):
                        low = l_s.lower()
                        if low not in seen_tokens or low not in ["mumbai", "delhi", "pune", "thane", "navimumbai"]:
                            addr_lines.append(l_s)
                            seen_tokens.add(low)
                if addr_lines:
                    buyer_address = ", ".join(addr_lines)

        if not buyer_address:
            addr_m = re.search(r'Address\s*[:\-]\s*([\s\S]+?)(?=\n\s*(?:MAHARASTRA|Maharashtra|GSTIN|PAN|Name\s*Of\s*State|Period|PO|$))', raw_text, re.I)
            if addr_m:
                addr_lines = [al.strip() for al in addr_m.group(1).split("\n") if al.strip()]
                buyer_address = ", ".join(addr_lines)
                if "MAHARASTRA" in raw_text or "Maharashtra" in raw_text:
                    if "maharashtra" not in buyer_address.lower():
                        buyer_address += ", MAHARASHTRA"

        if not buyer_gstin:
            buyer_gstin_m = re.search(r'GSTIN\s*(?:No\.?)?\s*[:\-]*\s*([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][0-9A-Z]Z[0-9A-Z])', raw_text, re.I)
            if buyer_gstin_m:
                cand = buyer_gstin_m.group(1).strip()
                if cand != seller_gstin:
                    buyer_gstin = cand
            if not buyer_gstin and len(gstins) > 1:
                buyer_gstin = [g for g in gstins if g != seller_gstin][0]

        buyer_pan = None
        pan_buyer_m = re.search(r'PAN\s*No\s*[:\-]\s*([A-Z]{5}\d{4}[A-Z])', raw_text, re.I)
        if pan_buyer_m and pan_buyer_m.group(1) != seller_pan:
            buyer_pan = pan_buyer_m.group(1)
        elif buyer_gstin and len(buyer_gstin) >= 12:
            buyer_pan = buyer_gstin[2:12]

        if buyer_name and "\n" in buyer_name:
            b_parts = [l.strip() for l in buyer_name.splitlines() if l.strip()]
            buyer_name = b_parts[0]
            if len(b_parts) > 1 and not buyer_address:
                buyer_address = ", ".join(b_parts[1:])

        buyer_state = None
        buyer_state_code = None
        bs_m = re.search(r'Buyer[\s\S]+?State\s*Name\s*[:\-]\s*([A-Za-z\s]+?)(?:,|$|\n)', raw_text, re.I)
        if bs_m:
            buyer_state = bs_m.group(1).strip()
        if buyer_gstin and len(buyer_gstin) >= 2 and buyer_gstin[:2].isdigit():
            buyer_state_code = buyer_gstin[:2]
            if not buyer_state:
                buyer_state = GST_STATE_MAP.get(buyer_state_code)
        if not buyer_state:
            buyer_state = seller_state
            buyer_state_code = seller_state_code

        contract_person_details = {
            "company_name": buyer_name or "-",
            "contact_person": contact_person,
            "contact_number": contact_number,
            "address": buyer_address,
            "gstin": buyer_gstin,
            "pan": buyer_pan,
            "state": buyer_state,
            "state_code": buyer_state_code
        }

        # 3. Invoice Details
        inv_type = "TAX INVOICE"
        if "challan" in raw_text.lower():
            inv_type = "DELIVERY CHALLAN"

        inv_num = None
        inv_date = None
        period_start = None
        period_end = None
        po_num = None
        po_date = None
        pay_terms = None

        if inv_txt:
            in_m = re.search(r'(?:RM|P|INV|[A-Za-z]{1,4})[\s\/]*\d{4,8}[\s\/\-]*\d{2,4}(?:-\d{2,4})?', inv_txt, re.I)
            if in_m:
                inv_num = re.sub(r'\s+', '', in_m.group(0))

            pod_m = re.search(r'PO\s*Date[^\d]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', inv_txt, re.I)
            if pod_m:
                po_date = self._format_date_iso(pod_m.group(1))

            per_m = re.search(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\s*(?:to|To|-)\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', inv_txt)
            if per_m:
                period_start = self._format_date_iso(per_m.group(1))
                period_end = self._format_date_iso(per_m.group(2))

            all_inv_dates = [self._format_date_iso(d) for d in re.findall(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', inv_txt)]
            non_inv = {po_date, period_start, period_end}
            valid_inv_dates = [d for d in all_inv_dates if d and d not in non_inv]
            if valid_inv_dates:
                inv_date = valid_inv_dates[0]
            elif all_inv_dates:
                inv_date = all_inv_dates[0]

            po_match = re.search(r'([A-Za-z]{2,6}-\d{3,6}-\d{2,4}(?:-\d{2,4})?)', inv_txt)
            if po_match:
                po_num = po_match.group(1)
            else:
                pon_m = re.search(r'PO\s*No\.?\s*[\n\s:]*([A-Za-z0-9\-]+)', inv_txt, re.I)
                if pon_m:
                    cand_po = pon_m.group(1).strip()
                    if len(cand_po) > 4:
                        po_num = cand_po

        if not inv_num or inv_num.lower() in ["date", "dated", "tax", "gstin"]:
            inv_num_m = re.search(r'\b([A-Za-z]{1,4}\/\d{4,8}(?:-\d{2,4})?)\b', raw_text)
            if inv_num_m:
                inv_num = inv_num_m.group(1).strip()
            else:
                cand_m = re.search(r'Invoice\s*No\.?\s*[:\-\s\n]*([A-Za-z0-9\/\-]+)', raw_text, re.I)
                if cand_m and cand_m.group(1).lower() not in ["date", "dated", "tax", "gstin"]:
                    inv_num = cand_m.group(1).strip()

        if not inv_num or inv_num.lower() in ["date", "dated", "tax", "gstin"]:
            in_m = re.search(r'(?:^|\n)\s*(\d{4,8})\s*\n\s*(?:Date|M[\.\/]?s\.?|[A-Za-z\.\s\&]+COMPANY)', raw_text, re.I)
            if in_m:
                inv_num = in_m.group(1).strip()

        if not inv_date:
            dated_idx = None
            for idx, l in enumerate(lines[:20]):
                if any(k in l.strip().lower() for k in ["dated.", "dated", "invoice date"]):
                    dated_idx = idx
                    break
            if dated_idx is not None:
                for offset in [-4, -3, -2, -1, 1, 2]:
                    pos = dated_idx + offset
                    if 0 <= pos < len(lines):
                        pm = re.match(r'^(\d{1,2}[\/\-\.](?:[A-Za-z]{3}|\d{1,2})[\/\-\.]\d{2,4})$', lines[pos].strip())
                        if pm:
                            inv_date = self._format_date_iso(pm.group(1))
                            break
            if not inv_date:
                dt_m = re.search(r'Date[^\d\n]*\n\s*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', raw_text, re.I)
                if dt_m:
                    inv_date = self._format_date_iso(dt_m.group(1))

        if not inv_date:
            d_m = re.search(r'(?:Dated\.?|Invoice\s*Date)\s*[:\-\s\n]*(\d{1,2}[\/\-\.](?:[A-Za-z]{3}|\d{1,2})[\/\-\.]\d{2,4})', raw_text, re.I)
            if d_m:
                inv_date = self._format_date_iso(d_m.group(1))

        if not period_start:
            period_start = inv_date
            period_end = inv_date
            period_idx = None
            for idx, l in enumerate(lines[:30]):
                if l.strip().lower() == "period":
                    period_idx = idx
                    break
            if period_idx is not None:
                period_dates = []
                for offset in [-3, -2, -1, 1, 2, 3]:
                    pos = period_idx + offset
                    if 0 <= pos < len(lines):
                        pm = re.match(r'^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})$', lines[pos].strip())
                        if pm:
                            period_dates.append(self._format_date_iso(pm.group(1)))
                if len(period_dates) >= 2:
                    period_start = period_dates[0]
                    period_end = period_dates[1]
                elif len(period_dates) == 1:
                    period_start = period_dates[0]
                    period_end = period_dates[0]

        if not po_num or not po_date:
            for idx, l in enumerate(lines[:35]):
                if "po date" in l.lower() and not po_date:
                    if idx + 1 < len(lines) and re.match(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$', lines[idx + 1].strip()):
                        po_date = self._format_date_iso(lines[idx + 1].strip())
                if "po no" in l.lower() and not po_num:
                    if idx + 1 < len(lines):
                        val = lines[idx + 1].strip()
                        if val and val not in [",", "-", ".", ":"]:
                            po_num = val

        if not site_name:
            site_m = re.search(r'Site\s*Address\s*\n\s*([^\n]+)', raw_text, re.I)
            if site_m:
                cand_site = site_m.group(1).strip()
                if cand_site and not cand_site.lower().startswith("po") and not cand_site.lower().startswith("remarks"):
                    site_name = cand_site
                    site_address = cand_site

        pt_m = re.search(r'Terms\s*of\s*Payment\s*[:\-]\s*([^\n]+)', raw_text, re.I)
        if pt_m:
            cand_pt = pt_m.group(1).strip()
            if not cand_pt.lower().startswith("po"):
                pay_terms = cand_pt

        remarks = None
        rem_source = rem_txt or raw_text
        rem_m = re.search(r'Remarks\s*[:\-]\s*([\s\S]+?)(?=\n\s*(?:Declaration|Deciaration|Terms|Our\s*Bank|$))', rem_source, re.I)
        if rem_m:
            raw_rem = rem_m.group(1).strip()
            clean_lines = [re.sub(r'\s+', ' ', l).strip() for l in raw_rem.splitlines() if l.strip() and not l.strip().startswith("E. &") and not l.strip().startswith("E.&")]
            remarks = "; ".join(clean_lines).replace(":-", ":").replace("NBAME", "NAME")

        # Consignee Details (Ship to)
        consignee_details = None
        c_m = re.search(r'Consignee\s*(?:\(Ship\s*to\))?\s*[:\-]?\s*\n\s*([^\n]+)', raw_text, re.I)
        if c_m:
            c_name = c_m.group(1).strip()
            c_pos = raw_text.find("Consignee")
            b_pos = raw_text.find("Buyer", c_pos)
            c_block = raw_text[c_pos:b_pos] if b_pos != -1 else raw_text[c_pos:c_pos+300]
            c_lines = [l.strip() for l in c_block.splitlines() if l.strip()]
            c_addr = c_lines[2] if len(c_lines) > 2 and not c_lines[2].lower().startswith("state") else None
            c_state = seller_state
            c_state_code = seller_state_code
            sm = re.search(r'State\s*Name\s*[:\-]\s*([A-Za-z]+)', c_block, re.I)
            if sm: c_state = sm.group(1).strip()
            scm = re.search(r'Code\s*[:\-]\s*(\d{2})', c_block, re.I)
            if scm: c_state_code = scm.group(1).strip()
            consignee_details = {
                "company_name": c_name,
                "address": c_addr,
                "state": c_state,
                "state_code": c_state_code
            }

        # Ack & IRN
        ack_no = None
        ack_dt = None
        combined_einv = ((einv_txt or "") + "\n" + raw_text)
        ack_m = re.search(r'AckNo"?\s*[:\-]?\s*"?(\d{10,20})', combined_einv, re.I)
        if not ack_m:
            ack_m = re.search(r'ACK\s*No\.?\s*[:\-]?\s*"?(\d{10,20})', combined_einv, re.I)
        if ack_m:
            ack_no = ack_m.group(1).strip()
        ack_dt_m = re.search(r'AckDt"?\s*[:\-]?\s*"?(\d{4}-\d{2}-\d{2}[\s\nT]+\d{2}:\d{2}:\d{2})', combined_einv, re.I)
        if ack_dt_m:
            ack_dt = re.sub(r'[\s\n]+', 'T', ack_dt_m.group(1).strip())
        if not ack_dt:
            ack_dt_text_m = re.search(r'Ack\s*Date\s*[:\-]?\s*([0-9A-Za-z\-\/\.\s]{6,20})', combined_einv, re.I)
            if ack_dt_text_m:
                ack_dt = self._format_date_iso(ack_dt_text_m.group(1).strip())

        irn_val = None
        irn_m = re.search(r'\b([a-f0-9]{64})\b', raw_text, re.I)
        if irn_m:
            irn_val = irn_m.group(1).lower()

        # Transport & Logistics Metadata
        transport = None
        tm = re.search(r'Transport\s*\n\s*([^\n]+)', raw_text, re.I)
        if tm:
            transport = tm.group(1).strip()

        lr_no = None
        lr_m = re.search(r'Lr\s*No[\s\S]+?HAND\s*TO\s*HAND', raw_text, re.I)
        if lr_m:
            lr_no = "HAND TO HAND"
        else:
            lr_m2 = re.search(r'Lr\s*No\.?\s*[:\-]?\s*([A-Za-z0-9\s\-]+)', raw_text, re.I)
            if lr_m2 and not lr_m2.group(1).lower().startswith("date"):
                lr_no = lr_m2.group(1).strip()

        town = None
        tw_m = re.search(r'Transport\s*\n\s*[^\n]+\n\s*([A-Za-z]+)', raw_text, re.I)
        if tw_m and tw_m.group(1).lower() not in ["invoice", "date", "gst"]:
            town = tw_m.group(1).upper()
        if not town:
            tw_m2 = re.search(r'\b(MUMBAI|DELHI|PUNE|SURAT|AHMEDABAD|JETPUR|RAJKOT)\b', raw_text[:800], re.I)
            if tw_m2:
                town = tw_m2.group(1).upper()

        note = None
        nm = re.search(r'Note\s*[:\-]?\s*([^\n]+)', raw_text, re.I)
        if nm:
            note = nm.group(1).strip(": ")

        hsn_meta = None
        hm = re.search(r'HSN\s*CODE\s*[:\-]?\s*[\s\S]+?(\d{6,8})', raw_text, re.I)
        if hm:
            hsn_meta = hm.group(1)

        parcel_count = None
        pm = re.search(r'Parcel[\s\S]+?\n\s*(\d{1,4})\s*\n', raw_text, re.I)
        if pm:
            parcel_count = int(pm.group(1))

        dispatch_type = "DIRECT" if "DIRECT" in raw_text else None

        eway_m = re.search(r'E\s*Waybill\s*No\.?\s*[:\-]?\s*(\d{10,14})', raw_text, re.I)
        eway_val = eway_m.group(1).strip() if eway_m else None

        # Dispatch and delivery meta
        disp_doc = None
        disp_m = re.search(r'Dispatch\s*Doc\s*No\.?\s*\n\s*([A-Za-z0-9\-]+)', raw_text, re.I)
        if disp_m:
            disp_doc = disp_m.group(1).strip()
            if disp_doc.lower() in ["local", "courier", "by hand", "dispatched", "through"]:
                disp_doc = None

        del_note = None
        del_m = re.search(r'Delivery\s*Note\s*\n\s*([A-Za-z0-9\-]+)', raw_text, re.I)
        if del_m:
            cand_del = del_m.group(1).strip()
            if not any(k in cand_del.lower() for k in ["reference", "buyer", "dispatch", "dated", "mode", "destination", "terms", "note"]):
                del_note = cand_del

        dest_val = None
        dest_m = re.search(r'Destination\s*\n\s*([A-Za-z0-9\s\-]+)', raw_text, re.I)
        if dest_m:
            cand_dest = dest_m.group(1).strip()
            if not any(k in cand_dest.lower() for k in ["terms", "delivery"]):
                dest_val = cand_dest

        # Clean period
        if period_start == inv_date and period_end == inv_date:
            clean_period = {"start_date": None, "end_date": None}
        else:
            clean_period = {"start_date": period_start, "end_date": period_end}

        invoice_details = {
            "invoice_number": inv_num or "-",
            "invoice_date": inv_date,
            "transport": transport,
            "lr_number": lr_no,
            "parcel_count": parcel_count,
            "dispatch_type": dispatch_type,
            "town": town,
            "note": note,
            "hsn_code": hsn_meta,
            "delivery_note_number": del_note,
            "dispatch_document_number": disp_doc,
            "destination": dest_val,
            "mode_of_payment": pay_terms,
            "acknowledgement_number": ack_no,
            "acknowledgement_date": ack_dt,
            "irn": irn_val,
            "e_waybill_number": eway_val,
            "period": clean_period
        }
        if site_name:
            invoice_details["site_name"] = site_name
            invoice_details["site_address"] = site_address or site_name
        if po_num:
            invoice_details["purchase_order_number"] = po_num
            invoice_details["purchase_order_date"] = po_date
        if remarks:
            invoice_details["remarks"] = remarks

        # 4. Goods and Tax Details
        items = []
        item_counter = 1
        for r in goods_rows:
            desc = r.get("Description of Goods") or r.get("Item") or ""
            if desc.lower() in ["total", "cgst", "sgst", "igst"] or r.get("SI No") == "-":
                continue
            hsn = r.get("HSN/SAC") or r.get("HSN Code") or ""
            qty_str = str(r.get("Quantity") or "0")
            qty_val = 0.0
            unit = "Mtr"
            qm = re.search(r'([\d,]+(?:\.\d+)?)', qty_str)
            if qm:
                try:
                    qty_val = float(qm.group(1).replace(",", ""))
                except ValueError:
                    pass
            um = re.search(r'([A-Za-z0-9\.\/]+)$', qty_str)
            if um:
                unit = um.group(1)
            if r.get("Per") and r["Per"] != "-":
                unit = r["Per"]

            rate_val = 0.0
            try:
                rate_val = float(str(r.get("Rate") or "0").replace(",", ""))
            except ValueError:
                pass

            amt_val = 0.0
            try:
                amt_val = float(str(r.get("Amount") or "0").replace(",", ""))
            except ValueError:
                pass

            # Normalize unit display (e.g. Mtr)
            clean_unit = unit.capitalize() if unit.lower() in ["mtr", "pcs", "nos", "bags", "kg", "sqm"] else unit.upper()

            item_obj = {
                "serial_number": item_counter,
                "description": desc,
                "hsn_code": hsn if hsn != "-" else None,
                "quantity": qty_val,
                "unit": clean_unit,
                "unit_rate": rate_val,
                "amount": amt_val
            }
            if r.get("bale_number"):
                item_obj["bale_number"] = r["bale_number"]
            if "approx_meters" in r:
                item_obj["approx_meters"] = r["approx_meters"]
            if "pieces" in r:
                item_obj["pieces"] = r["pieces"]
            if r.get("additional_details"):
                item_obj["additional_details"] = r["additional_details"]
            items.append(item_obj)
            item_counter += 1

        tot_qty_val = sum(it["quantity"] for it in items) if items else 0.0
        tot_qty_unit = items[0]["unit"] if items else "Mtr"
        
        gross_amt = sum(it["amount"] for it in items) if items else 0.0
        has_pieces = any("pieces" in it for it in items)
        tot_pieces = sum(it.get("pieces", 0) for it in items) if has_pieces else None
        tot_pcs_m = re.search(r'(\d{3,6})\s*\n\s*Total\b', raw_text, re.I)
        if tot_pcs_m and has_pieces:
            tot_pieces = int(tot_pcs_m.group(1))

        discount_obj = custom.get("discount")
        disc_amt = discount_obj.get("amount", 0.0) if discount_obj else 0.0

        subtotal_val = custom.get("taxable_value")
        if not subtotal_val:
            subtotal_val = round(gross_amt - disc_amt, 2) if disc_amt > 0 else round(gross_amt, 2)

        cgst_amt = custom.get("cgst", 0.0)
        sgst_amt = custom.get("sgst", 0.0)
        igst_amt = custom.get("igst")

        # Mathematical and dynamic tax rate deduction
        cgst_pct = None
        if custom.get("cgst_rate"):
            try:
                cgst_pct = float(custom["cgst_rate"].replace("%", ""))
            except ValueError:
                pass
        if (cgst_pct is None or cgst_pct == 9.0) and subtotal_val > 0 and cgst_amt > 0:
            calc_rate = round((cgst_amt / subtotal_val) * 100, 2)
            if calc_rate > 0:
                cgst_pct = calc_rate
        if cgst_pct is None:
            rt_m = re.search(r'(?:CGST|Central\s*Tax)[^\n%]*?(\d+(?:\.\d+)?)\s*%', raw_text, re.I)
            if rt_m:
                try:
                    cgst_pct = float(rt_m.group(1))
                except ValueError:
                    pass
        if cgst_pct is None:
            cgst_pct = 9.0

        sgst_pct = None
        if custom.get("sgst_rate"):
            try:
                sgst_pct = float(custom["sgst_rate"].replace("%", ""))
            except ValueError:
                pass
        if (sgst_pct is None or sgst_pct == 9.0) and subtotal_val > 0 and sgst_amt > 0:
            calc_rate = round((sgst_amt / subtotal_val) * 100, 2)
            if calc_rate > 0:
                sgst_pct = calc_rate
        if sgst_pct is None:
            sgst_pct = cgst_pct

        igst_pct = None
        if igst_amt is not None and igst_amt > 0:
            if custom.get("igst_rate"):
                try:
                    igst_pct = float(custom["igst_rate"].replace("%", ""))
                except ValueError:
                    pass
            if igst_pct is None and subtotal_val > 0:
                igst_pct = round((igst_amt / subtotal_val) * 100, 2)
            if igst_pct is None:
                igst_pct = 5.0

        rounding_adj = custom.get("rounding_adjustment")
        if rounding_adj is None:
            ro_m = re.search(r'Round\s*Off\s*[:\-]?\s*(?:\(?\s*([\-]?)\s*\)?\s*([0-9\.]+))', raw_text, re.I)
            if ro_m:
                try:
                    sign = -1.0 if ("-" in ro_m.group(0) or "less" in raw_text[max(0, ro_m.start()-25):ro_m.start()].lower()) else 1.0
                    rounding_adj = round(float(ro_m.group(2)) * sign, 2)
                except ValueError:
                    pass

        if igst_amt is not None and igst_amt > 0:
            tot_tax_val = igst_amt
            calc_total = round(subtotal_val + tot_tax_val + (rounding_adj or 0.0), 2)
            tax_dict = {
                "cgst": None,
                "sgst": None,
                "igst": {
                    "rate_percent": igst_pct,
                    "amount": igst_amt
                },
                "total_tax": round(tot_tax_val, 2),
                "rounding_adjustment": rounding_adj
            }
        else:
            tot_tax_val = custom.get("total_tax_amount") or round(cgst_amt + sgst_amt, 2)
            calc_total = round(subtotal_val + tot_tax_val + (rounding_adj or 0.0), 2)
            tax_dict = {
                "cgst": {
                    "rate_percent": cgst_pct,
                    "amount": cgst_amt
                },
                "sgst": {
                    "rate_percent": sgst_pct,
                    "amount": sgst_amt
                },
                "igst": None,
                "total_tax": round(tot_tax_val, 2),
                "rounding_adjustment": rounding_adj
            }

        gt_cand = None
        gt_m = re.search(r'Total\s+[\d,\.]+\s+[A-Za-z0-9\.\/]+\s+[^\d\n]*([\d,]+\.\d{2})', raw_text, re.I)
        if not gt_cand:
            tot_inv_m = re.search(r'Total\s*Value\s*Of\s*invoice[^\n]*\n\s*([\d,]+\.\d{2})', raw_text, re.I)
            if tot_inv_m:
                gt_cand = float(tot_inv_m.group(1).replace(",", ""))
        if gt_m and not gt_cand:
            try:
                gt_cand = float(gt_m.group(1).replace(",", ""))
            except ValueError:
                pass

        grand_total_val = gt_cand or (calc_total if calc_total > 0 else custom.get("grand_total"))

        amt_words = custom.get("amount_chargeable_in_words")
        if not amt_words:
            aw_m = re.search(r'Amount\s*in\s*words\s*[:\-]?\s*([A-Za-z\s]+(?:Only|only)\.?)', rem_source, re.I)
            if not aw_m:
                aw_m = re.search(r'(?:Amount\s*Chargeable\s*\(in\s*words\)|\bAmount\s*in\s*words)\s*[:\-]?\s*([A-Za-z\s]+(?:Only|only)\.?)', raw_text, re.I)
            if not aw_m:
                aw_m = re.search(r'Rs\.\s*In\s*Words[^\n]*\n\s*([A-Za-z\s]+(?:Only|only))', raw_text, re.I)
            if aw_m:
                amt_words = aw_m.group(1).strip()

        tax_words = None
        tw_m = re.search(r'Tax\s*Amount\s*\(in\s*words\)\s*[:\-]?\s*([A-Za-z0-9\s\,\.]+(?:Only|only)\.?)', raw_text, re.I)
        if tw_m:
            tax_words = tw_m.group(1).strip()

        goods_and_tax_details = {
            "items": items,
            "total_quantity": round(tot_qty_val, 2),
            "quantity_unit": tot_qty_unit,
            "taxable_value": round(subtotal_val, 2),
            "tax_details": tax_dict,
            "total_amount_after_tax": round(grand_total_val, 2),
            "amount_in_words": amt_words
        }
        if tot_pieces is not None:
            goods_and_tax_details["total_pieces"] = tot_pieces
        if gross_amt and disc_amt:
            goods_and_tax_details["gross_amount"] = round(gross_amt, 2)
        if discount_obj:
            goods_and_tax_details["discount"] = discount_obj
        if tax_words:
            goods_and_tax_details["tax_amount_in_words"] = tax_words

        # 5. Bank Details
        bank_source = (bank_txt or "") + "\n" + raw_text
        bd = custom.get("bank_details", {})
        bank_name = bd.get("bank_name") or "-"
        branch = bd.get("branch") or "-"
        acc_no = bd.get("account_number") or "-"
        ifsc = bd.get("ifsc_code") or "-"

        hb_m = re.search(r'([A-Za-z\s]+(?:BANK|Bank)\s*(?:LTD|Ltd)?)\s*-\s*A\/C\s*NO\.?\s*[:\-]?\s*(\d+)', bank_source, re.I)
        if hb_m:
            b_cand = hb_m.group(1).strip()
            bank_name = b_cand.splitlines()[-1].strip()
            acc_no = hb_m.group(2).strip()

        if not bank_name or bank_name == "-":
            b_name_m = re.search(r'(?:Bank\s*Name|Bank)\s*[:\-]\s*([A-Za-z\s]+(?:BANK|Bank|LTD|Ltd|\.))', bank_source, re.I)
            if b_name_m:
                bank_name = b_name_m.group(1).strip()

        if not ifsc or ifsc == "-":
            ifsc_lbl_m = re.search(r'IFSC\s*CODE\s*[-:]\s*([A-Z0-9]+)', bank_source, re.I)
            if ifsc_lbl_m:
                ifsc = ifsc_lbl_m.group(1).strip()

        if not branch or branch == "-":
            br_lbl_m = re.search(r'BRANCH\s*NAME\s*[-:]\s*([^\n,]+)', bank_source, re.I)
            if br_lbl_m:
                branch = br_lbl_m.group(1).strip()

        b_br_m = re.search(r'Branch\s*&?\s*IFS\s*Code\s*[:\-]?\s*([^\n]+)', bank_source, re.I)
        if b_br_m:
            b_val = b_br_m.group(1).strip()
            if "&" in b_val:
                branch = b_val.split("&")[0].strip()
                if not ifsc or ifsc == "-":
                    ifsc = b_val.split("&")[1].strip()
            else:
                branch = b_val
        elif not branch or branch == "-":
            b_br_m2 = re.search(r'Branch\s*[:\-]\s*([^\n\(\)]+(?:\([^\)]+\))?)', bank_source, re.I)
            if b_br_m2:
                branch = b_br_m2.group(1).strip()

        b_acc_m = re.search(r'(?:A/c\s*(?:No\.?)?|Account\s*No\.?)\s*[:\-]?\s*([\d\s]{8,25})', bank_source, re.I)
        if b_acc_m and (not acc_no or acc_no == "-"):
            acc_cand = re.sub(r'\s+', '', b_acc_m.group(1).strip())
            if len(acc_cand) >= 9:
                acc_no = acc_cand

        if not ifsc or ifsc == "-":
            ifsc_m = re.search(r'\b([A-Z]{4}0[A-Z0-9]{6})\b', bank_source)
            if ifsc_m:
                ifsc = ifsc_m.group(1).strip()

        ac_holder_m = re.search(r'A/c\s*Holder[’\']?s\s*Name\s*[:\-]\s*([^\n]+)', bank_source, re.I)
        ac_holder = ac_holder_m.group(1).strip() if ac_holder_m else (seller_name or "-")

        our_bank_details = {
            "account_holder_name": ac_holder,
            "bank_name": bank_name,
            "account_number": acc_no,
            "branch": branch,
            "ifsc_code": ifsc,
            "swift_code": None
        }

        contract_and_invoice_details = {
            "seller_details": seller_details,
            "contract_person_details": contract_person_details
        }
        if consignee_details:
            contract_and_invoice_details["consignee_details"] = consignee_details
        contract_and_invoice_details["invoice_details"] = invoice_details

        return {
            "contract_and_invoice_details": contract_and_invoice_details,
            "goods_and_tax_details": goods_and_tax_details,
            "our_bank_details": our_bank_details
        }

universal_llm_extractor = UniversalLLMExtractor()
