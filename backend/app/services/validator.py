import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DocumentValidator:
    def validate(self, normalized_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates structure, data types, missing fields, and financial totals.
        Never alters extracted values silently.
        """
        issues: List[Dict[str, Any]] = []
        status = "valid"

        # 1. Structural checks
        content = normalized_json.get("content", {})
        metadata = normalized_json.get("metadata", {})
        doc_type = normalized_json.get("document_type", "generic_document")

        if not content:
            issues.append({
                "field": "content",
                "issue_type": "missing_content",
                "message": "Document contains empty or unparsed content.",
                "severity": "warning"
            })
            status = "review_needed"

        # 2. Check for missing critical metadata in structured documents
        if doc_type in ["tax_invoice", "delivery_challan", "purchase_order"]:
            custom = content.get("custom_fields", {}) if isinstance(content, dict) else {}
            inv_details = custom.get("contract_and_invoice_details", {}).get("invoice_details", {}) if isinstance(custom, dict) else {}
            has_date = bool(metadata.get("dates")) or bool(inv_details.get("invoice_date")) or bool(custom.get("goods_and_tax_details", {}).get("acknowledgement_date"))
            if not has_date:
                issues.append({
                    "field": "metadata.dates",
                    "issue_type": "missing_date",
                    "message": "No invoice/challan date could be identified.",
                    "severity": "warning"
                })
            
            custom = content.get("custom_fields", {})
            g_and_t = custom.get("goods_and_tax_details", {})
            has_total = "grand_total" in custom or "total_amount" in custom or "total_amount_after_tax" in g_and_t
            if not has_total:
                issues.append({
                    "field": "content.custom_fields.grand_total",
                    "issue_type": "missing_total",
                    "message": "Document total amount was not explicitly identified.",
                    "severity": "warning"
                })

        # 3. Financial calculations check
        custom = content.get("custom_fields", {}) if isinstance(content, dict) else {}
        g_and_t = custom.get("goods_and_tax_details", {}) if isinstance(custom, dict) and isinstance(custom.get("goods_and_tax_details"), dict) else {}
        tax_details = g_and_t.get("tax_details", {}) if isinstance(g_and_t, dict) and isinstance(g_and_t.get("tax_details"), dict) else {}
        taxable = custom.get("taxable_value") or custom.get("assessable_value") or g_and_t.get("subtotal")

        cgst_info = tax_details.get("cgst") if isinstance(tax_details, dict) and isinstance(tax_details.get("cgst"), dict) else {}
        sgst_info = tax_details.get("sgst") if isinstance(tax_details, dict) and isinstance(tax_details.get("sgst"), dict) else {}
        igst_info = tax_details.get("igst") if isinstance(tax_details, dict) and isinstance(tax_details.get("igst"), dict) else {}

        cgst = custom.get("cgst") or cgst_info.get("amount")
        sgst = custom.get("sgst") or sgst_info.get("amount")
        igst = custom.get("igst") or igst_info.get("amount") or 0
        grand_total = custom.get("grand_total") or g_and_t.get("total_amount_after_tax")

        if taxable is not None and grand_total is not None:
            try:
                calc_total = float(taxable)
                if cgst is not None:
                    calc_total += float(cgst)
                if sgst is not None:
                    calc_total += float(sgst)
                if igst is not None:
                    calc_total += float(igst)
                if "rounding_adjustment" in tax_details and tax_details["rounding_adjustment"] is not None:
                    calc_total += float(tax_details["rounding_adjustment"])

                gt_float = float(grand_total)
                # Allow minor rounding discrepancy up to 1.0 currency unit
                if abs(calc_total - gt_float) > 1.0:
                    status = "calculation_mismatch"
                    issues.append({
                        "field": "custom_fields.grand_total",
                        "issue_type": "calculation_mismatch",
                        "message": (
                            f"Calculated sum of taxable value ({taxable}) + taxes ({cgst or 0} CGST, {sgst or 0} SGST, {igst or 0} IGST) "
                            f"= {calc_total:.2f}, which does not match extracted Grand Total of {gt_float:.2f}."
                        ),
                        "severity": "warning"
                    })
            except Exception as e:
                logger.debug(f"Numeric validation skipped due to parsing: {e}")

        # Check table headers vs row cells
        tables = content.get("tables", [])
        for t_idx, table in enumerate(tables):
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            if rows and not headers:
                issues.append({
                    "field": f"content.tables[{t_idx}]",
                    "issue_type": "missing_table_headers",
                    "message": f"Table on page {table.get('page', 1)} has rows but no extracted headers.",
                    "severity": "info"
                })

        if status == "valid" and any(i["severity"] == "warning" for i in issues):
            status = "review_needed"

        return {
            "status": status,
            "issues": issues
        }

document_validator = DocumentValidator()
