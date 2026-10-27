import React, { useState } from 'react';
import { ExternalLink, Copy, Check, Terminal, Code2 } from 'lucide-react';
import { copyTextToClipboard } from '../services/clipboard';

export default function ApiDocsView() {
  const [copiedKey, setCopiedKey] = useState(null);

  const copyCode = async (key, text) => {
    const success = await copyTextToClipboard(text);
    if (success) {
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 2000);
    }
  };

  const endpoints = [
    {
      key: 'upload',
      method: 'POST',
      path: '/api/v1/documents/extract',
      desc: 'Uploads a document (PDF, PNG, JPG), runs OCR/LLM extraction, validates structure, and persists normalized JSON in PostgreSQL.',
      curl: `curl -X POST "http://localhost:8000/api/v1/documents/extract" \\
  -F "file=@/path/to/invoice_or_contract.pdf" \\
  -F "async_mode=false"`,
      response: `{
  "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "completed",
  "message": "Document processed and extracted successfully."
}`
    },
    {
      key: 'get_json',
      method: 'GET',
      path: '/api/v1/documents/{document_id}',
      desc: 'Retrieves the complete normalized Universal JSON for any extracted document.',
      curl: `curl -X GET "http://localhost:8000/api/v1/documents/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"`,
      response: `{
  "document_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "schema_version": "1.0",
  "document_type": "tax_invoice",
  "source": { "filename": "sample_invoice.pdf", "file_type": "pdf", "page_count": 1 },
  "metadata": { "title": "Tax Invoice", "dates": ["18/09/2026"], "names": ["SAI SIYA INFRA", "Ananya Infrastructure"] },
  "content": {
    "tables": [...],
    "key_value_pairs": [...],
    "custom_fields": { "supplier_gstin": "27ANKPC6900E2ZA", "grand_total": 29205.00 }
  },
  "validation": { "status": "valid", "issues": [] }
}`
    },
    {
      key: 'list',
      method: 'GET',
      path: '/api/v1/documents?documentType=tax_invoice&status=completed&page=1&limit=20',
      desc: 'Search, paginate, and filter documents by type, filename, date ranges, or nested JSONB fields.',
      curl: `curl -X GET "http://localhost:8000/api/v1/documents?documentType=tax_invoice&status=completed&limit=10"`,
      response: `{
  "total": 42,
  "page": 1,
  "limit": 10,
  "documents": [...]
}`
    },
    {
      key: 'rag',
      method: 'POST',
      path: '/api/v1/documents/{document_id}/query',
      desc: 'Ask natural-language questions about document content with cited source pages and relevant excerpts.',
      curl: `curl -X POST "http://localhost:8000/api/v1/documents/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d/query" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What are the payment terms and due date?"}'`,
      response: `{
  "query": "What are the payment terms and due date?",
  "answer": "Payment is due within 30 days of receipt as stated on Page 2.",
  "citations": [
    { "page": 2, "content": "All invoices shall be paid within 30 days of receipt...", "score": 6.38 }
  ]
}`
    },
    {
      key: 'health',
      method: 'GET',
      path: '/api/v1/health',
      desc: 'Inspect backend connectivity, database status, Tesseract OCR availability, and active LLM model.',
      curl: `curl -X GET "http://localhost:8000/api/v1/health"`,
      response: `{
  "status": "healthy",
  "service": "Universal Document Intelligence Platform",
  "database": { "connected": true, "type": "PostgreSQL" },
  "ocr_engine": { "name": "Tesseract OCR", "available": true },
  "llm_provider": { "provider": "open_weight", "model": "qwen-2.5-72b-instruct" }
}`
    }
  ];

  return (
    <div className="animate-fade-in" style={{ maxWidth: 1000, margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 className="page-title">REST API Reference & Integration</h1>
          <p className="page-desc">
            Universal document extraction, search, raw text, and RAG retrieval endpoints.
          </p>
        </div>
        <a 
          href="http://localhost:8000/docs" 
          target="_blank" 
          rel="noopener noreferrer" 
          className="btn btn-primary"
        >
          Open Swagger UI <ExternalLink size={15} />
        </a>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
        {endpoints.map((ep) => (
          <div key={ep.key} className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
              <span 
                style={{ 
                  padding: '4px 10px', 
                  borderRadius: 6, 
                  fontWeight: 700, 
                  fontSize: '0.82rem',
                  background: ep.method === 'POST' ? 'rgba(99, 102, 241, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                  color: ep.method === 'POST' ? 'var(--primary)' : 'var(--success)'
                }}
              >
                {ep.method}
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, fontSize: '0.98rem' }}>
                {ep.path}
              </span>
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 16 }}>
              {ep.desc}
            </p>

            {/* cURL Example */}
            <div style={{ marginBottom: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Terminal size={13} /> Example Request (cURL)
                </span>
                <button 
                  className="btn btn-secondary" 
                  style={{ padding: '4px 8px', fontSize: '0.75rem' }}
                  onClick={() => copyCode(ep.key + '_curl', ep.curl)}
                >
                  {copiedKey === ep.key + '_curl' ? <Check size={12} color="var(--success)" /> : <Copy size={12} />} Copy
                </button>
              </div>
              <div className="code-block" style={{ fontSize: '0.8rem' }}>
                {ep.curl}
              </div>
            </div>

            {/* Response Example */}
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)', fontWeight: 600, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                <Code2 size={13} /> Sample JSON Response
              </div>
              <div className="code-block" style={{ fontSize: '0.78rem', maxHeight: 180, overflowY: 'auto' }}>
                {ep.response}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
