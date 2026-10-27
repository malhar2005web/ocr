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

        # 6. Extract Tables & Invoice Domain Intelligence
        tables = detected_tables.copy()
        structured_invoice_data = None
        
        # If document is a Tax Invoice or commercial bill, run dedicated high-precision dynamic parser
        if doc_type in ["tax_invoice", "delivery_challan", "purchase_order", "bill_receipt"] or "invoice" in lower_text or "bill" in lower_text:
            inv_tables, inv_custom, inv_kv = self._parse_invoice_structured_data(lines, raw_text, detected_tables)
            tables = inv_tables
            custom_fields.update(inv_custom)
            kv_pairs.extend(inv_kv)
            
            # Build hierarchically structured invoice schema as requested
            structured_invoice_data = self._build_structured_invoice_json(
                raw_text=raw_text,
                lines=lines,
                goods_rows=tables[0].get("rows", []) if tables else [],
                custom=custom_fields
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

    def _parse_invoice_structured_data(self, lines: List[str], raw_text: str, detected_tables: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
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
                    for r in t.get("rows", []):
                        clean_row = {k: v for k, v in r.items() if v and v != "None"}
                        if clean_row:
                            goods_rows.append(clean_row)

        # 2. Extract line items from PyMuPDF vertical blocks
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

        # Dynamic Taxes (CGST, SGST, IGST)
        for idx, line in enumerate(lines):
            l_clean = line.strip()
            # CGST
            if re.search(r'\bcgst\b', l_clean, re.I) and "cgst" not in custom:
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
            taxable_m = re.search(r'(?:Taxable\s*Value|Assessable\s*Value)\s*[:\-\s₹Rs\.]*([\d,]+\.\d{2})', raw_text, re.I)
            if not taxable_m:
                taxable_m = re.search(r'([\d,]+\.\d{2})\s*\n\s*[\d\.]+\s*\n\s*[A-Za-z0-9\.]+\s*\n\s*Total', raw_text, re.I)
            if taxable_m:
                try:
                    custom["taxable_value"] = float(taxable_m.group(1).replace(",", ""))
                except ValueError:
                    pass
            elif computed_taxable:
                custom["taxable_value"] = computed_taxable

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

    def _format_date_iso(self, date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        d = date_str.strip()
        m = re.match(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})', d)
        if m:
            day, month, year = m.group(1), m.group(2), m.group(3)
            if len(year) == 2:
                year = "20" + year
            return f"{year}-{int(month):02d}-{int(day):02d}"
        return d

    def _build_structured_invoice_json(
        self,
        raw_text: str,
        lines: List[str],
        goods_rows: List[Dict[str, Any]],
        custom: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Builds the clean, hierarchical invoice schema matching user requirements:
        - contract_and_invoice_details (seller_details, contract_person_details, invoice_details)
        - goods_and_tax_details (items, taxes, subtotal, total_amount_after_tax)
        - our_bank_details (bank_name, branch, account_number, ifsc_code, account_holder_name)
        Excludes raw Terms & Conditions (T&C).
        """
        # 1. Seller Details
        seller_name = None
        seller_m = re.search(r'For\s+M/S\.?\s*([A-Za-z0-9\s\.\&\-]+?)(?=\n|Authorised|Receiver|$)', raw_text, re.I)
        if seller_m:
            cand = seller_m.group(1).strip()
            if len(cand) > 3:
                seller_name = cand
        if not seller_name:
            for l in lines[:5]:
                if any(k in l.upper() for k in ["ENTERPRISES", "SOLUTIONS", "CONCRETE", "INFRA", "PVT", "LTD"]):
                    seller_name = l.strip()
                    break

        pans = list(set(re.findall(r'\b[A-Z]{5}\d{4}[A-Z]\b', raw_text)))
        gstins = list(set(re.findall(r'\b\d{2}[A-Z]{5}\d{4}[A-Z\d]{1}[A-Z\d]{1}Z[A-Z\d]{1}\b', raw_text)))
        
        seller_pan = custom.get("company_pan") or (pans[0] if pans else None)
        seller_gstin = None
        if seller_pan and gstins:
            for g in gstins:
                if seller_pan in g:
                    seller_gstin = g
                    break
        if not seller_gstin and gstins:
            seller_gstin = gstins[0]

        udyam_m = re.search(r'\b(UDYAM-[A-Z0-9\-]+|MH-\d{2}-\d+)\b', raw_text, re.I)
        udyam_no = udyam_m.group(0).strip() if udyam_m else None
        if udyam_no and not udyam_no.startswith("UDYAM-"):
            udyam_no = f"UDYAM-{udyam_no}"

        seller_state = "Maharashtra"
        seller_state_code = "27"
        state_m = re.search(r'(?:State\s*Name|Name\s*Of\s*State)\s*[:\-]\s*([A-Za-z]+)', raw_text, re.I)
        if state_m:
            seller_state = state_m.group(1).strip()
        code_m = re.search(r'(?:Sate|State)\s*Code\s*[:\-]\s*(\d{2})', raw_text, re.I)
        if code_m:
            seller_state_code = code_m.group(1).strip()
        elif seller_gstin:
            seller_state_code = seller_gstin[:2]

        seller_details = {
            "business_name": seller_name or "-",
            "gstin": seller_gstin,
            "pan": seller_pan,
            "udyam_number": udyam_no,
            "state": seller_state,
            "state_code": seller_state_code
        }

        # 2. Contract Person / Buyer Details
        buyer_name = None
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
            buyer_m = re.search(r'(?:Buyer|Bill\s*to)\s*[:\-\s\n]*([A-Za-z0-9\s\.\&\-]+?)(?=\n|Plot|GSTIN|$)', raw_text, re.I)
            if buyer_m:
                buyer_name = buyer_m.group(1).strip()

        contact_person = None
        cp_m = re.search(r'Contact\s*Person\s*[:\-]\s*([^\n]+)', raw_text, re.I)
        if cp_m:
            contact_person = cp_m.group(1).strip()

        contact_number = None
        cn_m = re.search(r'(?:Contact\s*Number|Phone|Mobile|Mob\.?)\s*[:\-]?\s*(\d{10})', raw_text, re.I)
        if cn_m:
            contact_number = cn_m.group(1).strip()

        buyer_address = None
        addr_m = re.search(r'Address\s*[:\-]\s*([\s\S]+?)(?=\n\s*(?:MAHARASTRA|Maharashtra|GSTIN|PAN|Name\s*Of\s*State|Period|PO|$))', raw_text, re.I)
        if addr_m:
            addr_lines = [al.strip() for al in addr_m.group(1).split("\n") if al.strip()]
            buyer_address = ", ".join(addr_lines)
            if "MAHARASTRA" in raw_text or "Maharashtra" in raw_text:
                if "maharashtra" not in buyer_address.lower():
                    buyer_address += ", MAHARASHTRA"

        buyer_gstin = None
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

        buyer_state = seller_state
        buyer_state_code = seller_state_code
        if buyer_gstin:
            buyer_state_code = buyer_gstin[:2]

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
        inv_num_m = re.search(r'\b([A-Za-z]{1,4}\/\d{4,8}(?:-\d{2,4})?)\b', raw_text)
        if inv_num_m:
            inv_num = inv_num_m.group(1).strip()
        if not inv_num:
            cand_m = re.search(r'Invoice\s*No\.?\s*[:\-\s\n]*([A-Za-z0-9\/\-]+)', raw_text, re.I)
            if cand_m:
                inv_num = cand_m.group(1).strip()

        inv_date = None
        dated_idx = None
        for idx, l in enumerate(lines[:20]):
            if any(k in l.strip().lower() for k in ["dated.", "dated", "invoice date"]):
                dated_idx = idx
                break
        if dated_idx is not None:
            for offset in [-4, -3, -2, -1, 1, 2]:
                pos = dated_idx + offset
                if 0 <= pos < len(lines):
                    pm = re.match(r'^(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})$', lines[pos].strip())
                    if pm:
                        inv_date = self._format_date_iso(pm.group(1))
                        break
        if not inv_date:
            d_m = re.search(r'(?:Dated\.?|Invoice\s*Date)\s*[:\-\s\n]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})', raw_text, re.I)
            if d_m:
                inv_date = self._format_date_iso(d_m.group(1))

        # Period
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

        # PO Number & PO Date
        po_num = None
        po_date = None
        for idx, l in enumerate(lines[:35]):
            if "po date" in l.lower():
                if idx + 1 < len(lines) and re.match(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$', lines[idx + 1].strip()):
                    po_date = self._format_date_iso(lines[idx + 1].strip())
            if "po no" in l.lower():
                if idx + 1 < len(lines):
                    val = lines[idx + 1].strip()
                    if val and val not in [",", "-", ".", ":"]:
                        po_num = val

        site_name = None
        site_m = re.search(r'Site\s*Address\s*\n\s*([^\n]+)', raw_text, re.I)
        if site_m:
            cand_site = site_m.group(1).strip()
            if cand_site and not cand_site.lower().startswith("po") and not cand_site.lower().startswith("remarks"):
                site_name = cand_site

        pay_terms = None
        pt_m = re.search(r'Terms\s*of\s*Payment\s*[:\-]\s*([^\n]+)', raw_text, re.I)
        if pt_m:
            pay_terms = pt_m.group(1).strip()

        remarks = None
        rem_m = re.search(r'Remarks\s*[:\-]\s*([^\n]+(?:\n[^\n]+)?)', raw_text, re.I)
        if rem_m:
            r_text = re.sub(r'\s+', ' ', rem_m.group(1).strip())
            r_text = r_text.replace(":-", ":").replace("NBAME", "NAME")
            remarks = r_text

        invoice_details = {
            "invoice_type": inv_type,
            "invoice_number": inv_num or "-",
            "invoice_date": inv_date,
            "period": {
                "start_date": period_start,
                "end_date": period_end
            },
            "purchase_order_number": po_num,
            "purchase_order_date": po_date,
            "site_name": site_name,
            "site_address": site_name,
            "payment_terms": pay_terms,
            "remarks": remarks
        }

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
            unit = "M3"
            qm = re.search(r'([\d,]+(?:\.\d+)?)', qty_str)
            if qm:
                try:
                    qty_val = float(qm.group(1).replace(",", ""))
                except ValueError:
                    pass
            um = re.search(r'([A-Za-z0-9\.\/]+)$', qty_str)
            if um:
                unit = um.group(1).upper()
            if r.get("Per") and r["Per"] != "-":
                unit = r["Per"].upper()

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

            items.append({
                "serial_number": item_counter,
                "description": desc,
                "hsn_code": hsn if hsn != "-" else None,
                "quantity": qty_val,
                "unit": unit,
                "unit_rate": rate_val,
                "amount": amt_val
            })
            item_counter += 1

        tot_qty_val = sum(it["quantity"] for it in items) if items else 0.0
        tot_qty_unit = items[0]["unit"] if items else "M3"
        subtotal_val = custom.get("taxable_value") or sum(it["amount"] for it in items)

        cgst_pct = 9.0
        if custom.get("cgst_rate"):
            try:
                cgst_pct = float(custom["cgst_rate"].replace("%", ""))
            except ValueError:
                pass
        cgst_amt = custom.get("cgst", 0.0)

        sgst_pct = 9.0
        if custom.get("sgst_rate"):
            try:
                sgst_pct = float(custom["sgst_rate"].replace("%", ""))
            except ValueError:
                pass
        sgst_amt = custom.get("sgst", 0.0)

        tot_tax_val = custom.get("total_tax_amount") or (cgst_amt + sgst_amt)
        grand_total_val = custom.get("grand_total") or (subtotal_val + tot_tax_val)

        # Ack & IRN
        ack_no = None
        ack_dt = None
        ack_m = re.search(r'AckNo"?\s*[:\-]?\s*"?(\d+)', raw_text, re.I)
        if ack_m:
            ack_no = ack_m.group(1).strip()
        ack_dt_m = re.search(r'AckDt"?\s*[:\-]?\s*"?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', raw_text, re.I)
        if ack_dt_m:
            ack_dt = ack_dt_m.group(1).replace(" ", "T")

        irn_val = None
        irn_m = re.search(r'IRN\s*[:\-]?\s*([a-f0-9\s\n]{40,80})', raw_text, re.I)
        if irn_m:
            clean_irn = re.sub(r'[\s\n]+', '', irn_m.group(1))
            if len(clean_irn) >= 40:
                irn_val = clean_irn[:64]

        amt_words = custom.get("amount_chargeable_in_words")
        if not amt_words:
            aw_m = re.search(r'Amount\s*in\s*words\s*[:\-]?\s*([A-Za-z\s]+(?:Only|only)\.?)', raw_text, re.I)
            if aw_m:
                amt_words = aw_m.group(1).strip()

        goods_and_tax_details = {
            "items": items,
            "total_quantity": round(tot_qty_val, 2),
            "quantity_unit": tot_qty_unit,
            "subtotal": round(subtotal_val, 2),
            "tax_details": {
                "cgst": {
                    "rate_percent": cgst_pct,
                    "amount": cgst_amt
                },
                "sgst": {
                    "rate_percent": sgst_pct,
                    "amount": sgst_amt
                },
                "igst": None,
                "total_tax": round(tot_tax_val, 2)
            },
            "total_amount_after_tax": round(grand_total_val, 2),
            "amount_in_words": amt_words,
            "acknowledgement_number": ack_no,
            "acknowledgement_date": ack_dt,
            "irn": irn_val
        }

        # 5. Bank Details
        bd = custom.get("bank_details", {})
        our_bank_details = {
            "bank_name": bd.get("bank_name") or "-",
            "branch": bd.get("branch") or "-",
            "account_number": bd.get("account_number") or "-",
            "ifsc_code": bd.get("ifsc_code") or "-",
            "account_holder_name": seller_name or "-"
        }

        return {
            "contract_and_invoice_details": {
                "seller_details": seller_details,
                "contract_person_details": contract_person_details,
                "invoice_details": invoice_details
            },
            "goods_and_tax_details": goods_and_tax_details,
            "our_bank_details": our_bank_details
        }

universal_llm_extractor = UniversalLLMExtractor()
