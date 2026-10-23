import React, { useState, useEffect } from 'react';
import { FileText, CheckCircle2, AlertTriangle, XCircle, ArrowUpRight, Cpu, Database, ScanText, RefreshCw } from 'lucide-react';
import { api } from '../services/api';

export default function DashboardView({ onNavigate, onSelectDocument }) {
  const [metrics, setMetrics] = useState({ total: 0, completed: 0, review: 0, failed: 0 });
  const [recentDocs, setRecentDocs] = useState([]);
  const [systemHealth, setSystemHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadData = async () => {
    setLoading(true);
    try {
      const [listRes, healthRes] = await Promise.all([
        api.listDocuments({ limit: 8, sortBy: 'created_at', sortOrder: 'DESC' }),
        api.getHealth().catch(() => null)
      ]);

      const docs = listRes.documents || [];
      setRecentDocs(docs);
      setSystemHealth(healthRes);

      // Compute statistics
      const total = listRes.total || docs.length;
      const completed = docs.filter(d => d.status === 'completed' || d.status === 'valid').length;
      const review = docs.filter(d => d.status === 'review_needed' || d.status === 'calculation_mismatch').length;
      const failed = docs.filter(d => d.status === 'failed').length;
      setMetrics({ total, completed, review, failed });
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="animate-fade-in">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 className="page-title">Executive Dashboard</h1>
          <p className="page-desc">Overview of ingested documents, extraction metrics, and platform status.</p>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn btn-secondary" onClick={loadData}>
            <RefreshCw size={16} /> Refresh
          </button>
          <button className="btn btn-primary" onClick={() => onNavigate('upload')}>
            <FileText size={16} /> Extract Document
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="metrics-grid">
        <div className="card metric-card">
          <div className="metric-icon-wrap" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#6366f1' }}>
            <FileText size={28} />
          </div>
          <div>
            <div className="metric-val">{metrics.total}</div>
            <div className="metric-label">Total Documents Ingested</div>
          </div>
        </div>

        <div className="card metric-card">
          <div className="metric-icon-wrap" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981' }}>
            <CheckCircle2 size={28} />
          </div>
          <div>
            <div className="metric-val">{metrics.completed}</div>
            <div className="metric-label">Extracted & Verified</div>
          </div>
        </div>

        <div className="card metric-card">
          <div className="metric-icon-wrap" style={{ background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b' }}>
            <AlertTriangle size={28} />
          </div>
          <div>
            <div className="metric-val">{metrics.review}</div>
            <div className="metric-label">Awaiting User Review</div>
          </div>
        </div>

        <div className="card metric-card">
          <div className="metric-icon-wrap" style={{ background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444' }}>
            <XCircle size={28} />
          </div>
          <div>
            <div className="metric-val">{metrics.failed}</div>
            <div className="metric-label">Processing Errors</div>
          </div>
        </div>
      </div>

      {/* System Infrastructure Status */}
      {systemHealth && (
        <div className="card" style={{ marginBottom: 32, padding: '20px 24px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: 14, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Active Platform Engines & Storage
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Database size={20} color={systemHealth.database?.connected ? 'var(--success)' : 'var(--warning)'} />
              <div>
                <div style={{ fontSize: '0.88rem', fontWeight: 600 }}>PostgreSQL Storage</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {systemHealth.database?.connected ? 'Connected (Host: ' + systemHealth.database?.host + ')' : 'Fallback Storage Active'}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <ScanText size={20} color={systemHealth.ocr_engine?.available ? 'var(--success)' : 'var(--info)'} />
              <div>
                <div style={{ fontSize: '0.88rem', fontWeight: 600 }}>OCR Vision Engine</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {systemHealth.ocr_engine?.available ? 'Tesseract Ready' : 'PyMuPDF Native Text'}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Cpu size={20} color={systemHealth.llm_provider?.configured ? 'var(--success)' : 'var(--info)'} />
              <div>
                <div style={{ fontSize: '0.88rem', fontWeight: 600 }}>Inference Engine</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {systemHealth.llm_provider?.configured ? `Open-Weight (${systemHealth.llm_provider?.model})` : 'Heuristic OCR Engine Active'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent Activity Section */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 style={{ fontSize: '1.2rem', fontWeight: 600 }}>Recently Processed Documents</h2>
          <button 
            className="btn btn-secondary" 
            style={{ fontSize: '0.82rem', padding: '6px 12px' }}
            onClick={() => onNavigate('explorer')}
          >
            View All Documents <ArrowUpRight size={14} />
          </button>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-muted)' }}>
            Loading document feed...
          </div>
        ) : recentDocs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '48px 0' }}>
            <FileText size={48} color="var(--text-dim)" style={{ marginBottom: 12 }} />
            <div style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-muted)' }}>No documents uploaded yet</div>
            <p style={{ fontSize: '0.88rem', color: 'var(--text-dim)', marginTop: 4, marginBottom: 16 }}>
              Upload an invoice, delivery challan, contract, or image to see extraction results.
            </p>
            <button className="btn btn-primary" onClick={() => onNavigate('upload')}>
              Upload Your First Document
            </button>
          </div>
        ) : (
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Type</th>
                  <th>Status</th>
                  <th>Pages</th>
                  <th>Size</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {recentDocs.map((doc) => (
                  <tr key={doc.id}>
                    <td style={{ fontWeight: 600 }}>{doc.filename}</td>
                    <td>
                      <span style={{ textTransform: 'capitalize', color: 'var(--text-muted)' }}>
                        {(doc.document_type || 'unclassified').replace('_', ' ')}
                      </span>
                    </td>
                    <td>
                      <span className={`badge badge-${doc.status}`}>
                        {doc.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td>{doc.page_count || 1}</td>
                    <td style={{ color: 'var(--text-muted)' }}>
                      {((doc.file_size || 0) / 1024).toFixed(1)} KB
                    </td>
                    <td>
                      <button 
                        className="btn btn-secondary"
                        style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                        onClick={() => onSelectDocument(doc.id)}
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
