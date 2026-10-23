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
            if not metadata.get("dates"):
                issues.append({
                    "field": "metadata.dates",
                    "issue_type": "missing_date",
                    "message": "No invoice/challan date could be identified.",
                    "severity": "warning"
                })
            
            custom = content.get("custom_fields", {})
            if "grand_total" not in custom and "total_amount" not in custom:
                issues.append({
                    "field": "content.custom_fields.grand_total",
                    "issue_type": "missing_total",
                    "message": "Document total amount was not explicitly identified.",
                    "severity": "warning"
                })

        # 3. Financial calculations check
        custom = content.get("custom_fields", {})
        taxable = custom.get("taxable_value") or custom.get("assessable_value")
        cgst = custom.get("cgst")
        sgst = custom.get("sgst")
        igst = custom.get("igst", 0)
        grand_total = custom.get("grand_total")

        if taxable is not None and grand_total is not None:
            try:
                calc_total = float(taxable)
                if cgst is not None:
                    calc_total += float(cgst)
                if sgst is not None:
                    calc_total += float(sgst)
                if igst is not None:
                    calc_total += float(igst)

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
