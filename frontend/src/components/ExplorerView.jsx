import React, { useState, useEffect } from 'react';
import { Search, Filter, Eye, Download, Calendar, RefreshCw, ChevronLeft, ChevronRight, FileJson } from 'lucide-react';
import { api } from '../services/api';

export default function ExplorerView({ onSelectDocument }) {
  const [documents, setDocuments] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);

  // Filters
  const [filename, setFilename] = useState('');
  const [documentType, setDocumentType] = useState('');
  const [status, setStatus] = useState('');
  const [fromDate, setFromDate] = useState('');
  const [toDate, setToDate] = useState('');
  const [page, setPage] = useState(1);
  const [limit] = useState(15);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const res = await api.listDocuments({
        filename: filename || undefined,
        documentType: documentType || undefined,
        status: status || undefined,
        fromDate: fromDate || undefined,
        toDate: toDate || undefined,
        page,
        limit,
        sortBy: 'created_at',
        sortOrder: 'DESC'
      });
      setDocuments(res.documents || []);
      setTotal(res.total || 0);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [page, documentType, status]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    fetchDocuments();
  };

  const handleExportJSON = (doc) => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(doc.normalized_json || doc, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `${doc.filename || 'document'}_extracted.json`);
    dlAnchor.click();
  };

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="animate-fade-in">
      <div className="page-header">
        <h1 className="page-title">Document Intelligence Explorer</h1>
        <p className="page-desc">Query, search, and filter structured records and normalized JSON across all processed files.</p>
      </div>

      {/* Filter Toolbar */}
      <div className="card" style={{ marginBottom: 24, padding: '18px 24px' }}>
        <form onSubmit={handleSearchSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14, alignItems: 'center' }}>
          {/* Filename Search */}
          <div style={{ position: 'relative' }}>
            <input 
              type="text" 
              className="form-input" 
              placeholder="Search filename..."
              value={filename}
              onChange={(e) => setFilename(e.target.value)}
              style={{ paddingLeft: 34 }}
            />
            <Search size={16} color="var(--text-dim)" style={{ position: 'absolute', left: 10, top: 12 }} />
          </div>

          {/* Document Type Filter */}
          <select 
            className="form-select"
            value={documentType}
            onChange={(e) => { setDocumentType(e.target.value); setPage(1); }}
          >
            <option value="">All Document Types</option>
            <option value="tax_invoice">Tax Invoice</option>
            <option value="delivery_challan">Delivery Challan</option>
            <option value="purchase_order">Purchase Order</option>
            <option value="contract_agreement">Contract / Agreement</option>
            <option value="bank_statement">Bank Statement</option>
            <option value="salary_slip">Salary Slip</option>
            <option value="generic_document">Generic Document</option>
          </select>

          {/* Status Filter */}
          <select 
            className="form-select"
            value={status}
            onChange={(e) => { setStatus(e.target.value); setPage(1); }}
          >
            <option value="">All Statuses</option>
            <option value="completed">Completed / Valid</option>
            <option value="review_needed">Review Needed</option>
            <option value="processing">Processing</option>
            <option value="failed">Failed</option>
          </select>

          {/* Date Filter */}
          <div style={{ display: 'flex', gap: 8 }}>
            <input 
              type="date" 
              className="form-input" 
              value={fromDate}
              onChange={(e) => setFromDate(e.target.value)}
              title="From Date"
            />
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="submit" className="btn btn-primary" style={{ flex: 1 }}>
              <Filter size={15} /> Apply
            </button>
            <button 
              type="button" 
              className="btn btn-secondary"
              onClick={() => { setFilename(''); setDocumentType(''); setStatus(''); setFromDate(''); setToDate(''); setPage(1); fetchDocuments(); }}
            >
              <RefreshCw size={15} />
            </button>
          </div>
        </form>
      </div>

      {/* Results Table */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            Showing {documents.length} of {total} documents
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)' }}>
            Loading documents...
          </div>
        ) : documents.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 0', color: 'var(--text-muted)' }}>
            No documents matched your query. Try clearing filters.
          </div>
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Document Type</th>
                  <th>Status</th>
                  <th>Pages</th>
                  <th>Date Ingested</th>
                  <th style={{ textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.id}>
                    <td style={{ fontWeight: 600 }}>{doc.filename}</td>
                    <td>
                      <span style={{ textTransform: 'capitalize', color: '#cbd5e1' }}>
                        {(doc.document_type || 'unclassified').replace('_', ' ')}
                      </span>
                    </td>
                    <td>
                      <span className={`badge badge-${doc.status}`}>
                        {doc.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td>{doc.page_count || 1}</td>
                    <td style={{ color: 'var(--text-dim)', fontSize: '0.82rem' }}>
                      {doc.created_at ? new Date(doc.created_at).toLocaleString() : 'N/A'}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: 8 }}>
                        <button 
                          className="btn btn-secondary" 
                          style={{ padding: '6px 10px', fontSize: '0.8rem' }}
                          title="Export JSON"
                          onClick={() => handleExportJSON(doc)}
                        >
                          <FileJson size={14} />
                        </button>
                        <a 
                          href={api.getDownloadUrl(doc.id)} 
                          className="btn btn-secondary" 
                          style={{ padding: '6px 10px', fontSize: '0.8rem' }}
                          title="Download Original File"
                          download
                        >
                          <Download size={14} />
                        </a>
                        <button 
                          className="btn btn-primary" 
                          style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                          onClick={() => onSelectDocument(doc.id)}
                        >
                          <Eye size={14} /> Inspect
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 20 }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Page {page} of {totalPages}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button 
              className="btn btn-secondary" 
              disabled={page <= 1}
              onClick={() => setPage(p => Math.max(1, p - 1))}
              style={{ opacity: page <= 1 ? 0.5 : 1 }}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <button 
              className="btn btn-secondary" 
              disabled={page >= totalPages}
              onClick={() => setPage(p => p + 1)}
              style={{ opacity: page >= totalPages ? 0.5 : 1 }}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
