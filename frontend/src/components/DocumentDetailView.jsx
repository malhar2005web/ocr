import React, { useState, useEffect } from 'react';
import { 
  ArrowLeft, Download, Copy, Check, FileText, Code, AlignLeft, 
  HelpCircle, ShieldCheck, AlertTriangle, Table as TableIcon, 
  ExternalLink, Send, Bot, CheckCircle2, ChevronDown
} from 'lucide-react';
import { api } from '../services/api';

export default function DocumentDetailView({ documentId, onBack }) {
  const [doc, setDoc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('structured'); // structured, json, raw, rag, validation
  const [copied, setCopied] = useState(false);

  // RAG state
  const [ragQuery, setRagQuery] = useState('');
  const [ragLoading, setRagLoading] = useState(false);
  const [ragMessages, setRagMessages] = useState([]);

  useEffect(() => {
    const loadDoc = async () => {
      setLoading(true);
      try {
        const data = await api.getDocument(documentId);
        setDoc(data);
      } catch (err) {
        console.error('Failed to load document details:', err);
      } finally {
        setLoading(false);
      }
    };
    loadDoc();
  }, [documentId]);

  const handleCopyJSON = () => {
    if (!doc) return;
    navigator.clipboard.writeText(JSON.stringify(doc, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadJSON = () => {
    if (!doc) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(doc, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', `${doc.source?.filename || 'document'}_normalized.json`);
    dlAnchor.click();
  };

  const handleSendRag = async (e) => {
    e.preventDefault();
    if (!ragQuery.trim() || ragLoading) return;

    const userText = ragQuery;
    setRagQuery('');
    setRagMessages(prev => [...prev, { role: 'user', text: userText }]);
    setRagLoading(true);

    try {
      const res = await api.queryDocumentRAG(documentId, userText);
      setRagMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          text: res.answer, 
          citations: res.citations || [] 
        }
      ]);
    } catch (err) {
      setRagMessages(prev => [
        ...prev, 
        { 
          role: 'assistant', 
          text: `Query error: ${err.message || 'Failed to analyze document'}`, 
          citations: [] 
        }
      ]);
    } finally {
      setRagLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0', color: 'var(--text-muted)' }}>
        Loading document intelligence analysis...
      </div>
    );
  }

  if (!doc) {
    return (
      <div style={{ textAlign: 'center', padding: '80px 0' }}>
        <h2>Document not found or extraction pending.</h2>
        <button className="btn btn-primary" onClick={onBack} style={{ marginTop: 16 }}>
          <ArrowLeft size={16} /> Back to Explorer
        </button>
      </div>
    );
  }

  const { source, metadata, content, raw_text, extraction_metadata, validation, document_type } = doc;
  const isPdf = source?.file_type?.toLowerCase() === 'pdf';
  const downloadUrl = api.getDownloadUrl(documentId);

  return (
    <div className="animate-fade-in">
      {/* Top Bar Navigation */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <button className="btn btn-secondary" onClick={onBack}>
          <ArrowLeft size={16} /> Back
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span className={`badge badge-${validation?.status || 'completed'}`}>
            {validation?.status?.replace('_', ' ') || 'Completed'}
          </span>
          <button className="btn btn-secondary" onClick={handleDownloadJSON}>
            <Download size={15} /> Download JSON
          </button>
          <a href={downloadUrl} className="btn btn-primary" download>
            <Download size={15} /> Download Original
          </a>
        </div>
      </div>

      {/* Split Layout: Left Preview, Right Intelligence View */}
      <div className="detail-split-view">
        {/* Left Column: Original File Preview */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(15, 23, 42, 0.6)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontWeight: 600, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: 8 }}>
              <FileText size={16} color="var(--primary)" />
              {source?.filename}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
              {source?.page_count} {source?.page_count === 1 ? 'Page' : 'Pages'} • {source?.file_type?.toUpperCase()}
            </div>
          </div>

          <div style={{ flex: 1, background: 'var(--bg-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 650, borderRadius: 'var(--radius-md)' }}>
            {isPdf ? (
              <iframe 
                src={`${downloadUrl}#toolbar=0`} 
                title="Document Preview"
                style={{ width: '100%', height: '100%', minHeight: 650, border: 'none' }}
              />
            ) : (
              <img 
                src={downloadUrl} 
                alt="Uploaded Document"
                style={{ maxWidth: '100%', maxHeight: 650, objectFit: 'contain', padding: 16 }}
              />
            )}
          </div>
        </div>

        {/* Right Column: Normalized Structured Intelligence */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          {/* Tabs */}
          <div className="tab-list">
            <button 
              className={`tab-btn ${activeTab === 'structured' ? 'active' : ''}`}
              onClick={() => setActiveTab('structured')}
            >
              <FileText size={15} style={{ verticalAlign: 'middle', marginRight: 6 }} /> Structured
            </button>
            <button 
              className={`tab-btn ${activeTab === 'json' ? 'active' : ''}`}
              onClick={() => setActiveTab('json')}
            >
              <Code size={15} style={{ verticalAlign: 'middle', marginRight: 6 }} /> Universal JSON
            </button>
            <button 
              className={`tab-btn ${activeTab === 'rag' ? 'active' : ''}`}
              onClick={() => setActiveTab('rag')}
            >
              <Bot size={15} style={{ verticalAlign: 'middle', marginRight: 6 }} /> Ask AI (RAG)
            </button>
            <button 
              className={`tab-btn ${activeTab === 'raw' ? 'active' : ''}`}
              onClick={() => setActiveTab('raw')}
            >
              <AlignLeft size={15} style={{ verticalAlign: 'middle', marginRight: 6 }} /> Raw Text
            </button>
            <button 
              className={`tab-btn ${activeTab === 'validation' ? 'active' : ''}`}
              onClick={() => setActiveTab('validation')}
            >
              <ShieldCheck size={15} style={{ verticalAlign: 'middle', marginRight: 6 }} /> Validation
            </button>
          </div>

          {/* Tab 1: Structured View */}
          {activeTab === 'structured' && (
            <div style={{ overflowY: 'auto', flex: 1, paddingRight: 4 }}>
              {/* Validation Warning Alert if any */}
              {validation?.issues?.length > 0 && (
                <div style={{ padding: '12px 16px', background: 'var(--warning-bg)', border: '1px solid rgba(245, 158, 11, 0.3)', borderRadius: 'var(--radius-md)', marginBottom: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--warning)', fontWeight: 600, fontSize: '0.88rem' }}>
                    <AlertTriangle size={16} /> Validation Notice:
                  </div>
                  <ul style={{ margin: '6px 0 0 24px', fontSize: '0.82rem', color: 'var(--warning)' }}>
                    {validation.issues.map((iss, idx) => (
                      <li key={idx}>{iss.message}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Document Overview Summary Cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 24 }}>
                <div style={{ padding: '12px 16px', background: 'var(--bg-subtle)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Classification</div>
                  <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-main)', textTransform: 'capitalize' }}>
                    {document_type.replace('_', ' ')}
                  </div>
                </div>

                {content?.custom_fields?.grand_total && (
                  <div style={{ padding: '12px 16px', background: 'var(--success-bg)', borderRadius: 8, border: '1px solid var(--success-border)' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--success)', textTransform: 'uppercase' }}>Grand Total</div>
                    <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-main)' }}>
                      ₹{Number(content.custom_fields.grand_total).toLocaleString('en-IN')}
                    </div>
                  </div>
                )}

                {metadata?.dates?.length > 0 && (
                  <div style={{ padding: '12px 16px', background: 'var(--bg-subtle)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', textTransform: 'uppercase' }}>Detected Dates</div>
                    <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-main)' }}>
                      {metadata.dates.join(', ')}
                    </div>
                  </div>
                )}
              </div>

              {/* Key Value Pairs */}
              {content?.key_value_pairs?.length > 0 && (
                <div style={{ marginBottom: 28 }}>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 600, marginBottom: 12, color: 'var(--text-muted)' }}>
                    Key-Value Fields ({content.key_value_pairs.length})
                  </h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                    {content.key_value_pairs.map((kv, idx) => (
                      <div key={idx} style={{ padding: '8px 12px', background: 'var(--bg-subtle)', borderRadius: 6, border: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)', fontWeight: 500 }}>{kv.key}</div>
                        <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-main)', wordBreak: 'break-word' }}>
                          {String(kv.value)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Tables */}
              {content?.tables?.length > 0 && (
                <div style={{ marginBottom: 28 }}>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 600, marginBottom: 12, color: 'var(--text-muted)' }}>
                    Structured Tables ({content.tables.length})
                  </h3>
                  {content.tables.map((tbl, tIdx) => (
                    <div key={tIdx} className="table-container" style={{ marginBottom: 16 }}>
                      <table className="data-table">
                        <thead>
                          <tr>
                            {tbl.headers?.map((h, hIdx) => (
                              <th key={hIdx}>{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {tbl.rows?.map((row, rIdx) => (
                            <tr key={rIdx}>
                              {tbl.headers?.map((h, cIdx) => (
                                <td key={cIdx}>{row[h] || '-'}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ))}
                </div>
              )}

              {/* Custom / Dynamic Fields */}
              {content?.custom_fields && Object.keys(content.custom_fields).length > 0 && (
                <div>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 600, marginBottom: 12, color: 'var(--text-muted)' }}>
                    Extracted Domain Fields
                  </h3>
                  <div className="code-block" style={{ fontSize: '0.82rem' }}>
                    {JSON.stringify(content.custom_fields, null, 2)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Tab 2: Universal JSON */}
          {activeTab === 'json' && (
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginBottom: 8 }}>
                <button className="btn btn-secondary" style={{ padding: '6px 12px', fontSize: '0.8rem' }} onClick={handleCopyJSON}>
                  {copied ? <Check size={14} color="var(--success)" /> : <Copy size={14} />} {copied ? 'Copied' : 'Copy JSON'}
                </button>
              </div>
              <div className="code-block" style={{ flex: 1, maxHeight: 580, overflowY: 'auto' }}>
                {JSON.stringify(doc, null, 2)}
              </div>
            </div>
          )}

          {/* Tab 3: Ask Document (RAG) */}
          {activeTab === 'rag' && (
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 12 }}>
                Ask natural language questions about this specific document. The AI retrieves cited excerpts directly from source pages.
              </div>

              {/* Chat Thread */}
              <div style={{ flex: 1, minHeight: 380, maxHeight: 420, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, padding: '12px 4px' }}>
                {ragMessages.length === 0 ? (
                  <div style={{ textAlign: 'center', margin: 'auto', color: 'var(--text-dim)', fontSize: '0.88rem' }}>
                    Try asking:
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center', marginTop: 12 }}>
                      {['What is the total amount?', 'Who are the parties involved?', 'What payment terms are stated?', 'What items are listed?'].map((sample, i) => (
                        <button 
                          key={i} 
                          className="btn btn-secondary" 
                          style={{ fontSize: '0.78rem', padding: '6px 10px' }}
                          onClick={() => setRagQuery(sample)}
                        >
                          "{sample}"
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  ragMessages.map((msg, idx) => (
                    <div 
                      key={idx} 
                      style={{
                        alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        maxWidth: '85%',
                        padding: '12px 16px',
                        borderRadius: 'var(--radius-md)',
                        background: msg.role === 'user' ? 'var(--tiffany)' : 'var(--bg-subtle)',
                        border: '1px solid var(--border-subtle)',
                        color: msg.role === 'user' ? 'var(--text-on-primary)' : 'var(--text-main)',
                        fontSize: '0.88rem'
                      }}
                    >
                      <div style={{ fontWeight: 600, fontSize: '0.75rem', marginBottom: 4, opacity: 0.8 }}>
                        {msg.role === 'user' ? 'You' : 'DocuIntel Assistant'}
                      </div>
                      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
                      
                      {msg.citations?.length > 0 && (
                        <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px solid rgba(255, 255, 255, 0.1)', fontSize: '0.75rem' }}>
                          <span style={{ fontWeight: 600, color: 'var(--primary)' }}>Citations:</span>
                          {msg.citations.map((c, cIdx) => (
                            <div key={cIdx} style={{ color: 'var(--text-muted)', marginTop: 2 }}>
                              • Page {c.page}: "{c.content}"
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  ))
                )}
                {ragLoading && (
                  <div style={{ color: 'var(--primary)', fontSize: '0.85rem' }}>
                    Analyzing document context...
                  </div>
                )}
              </div>

              {/* Input Form */}
              <form onSubmit={handleSendRag} style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                <input 
                  type="text" 
                  className="form-input" 
                  placeholder="Ask a question about this document..."
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                />
                <button type="submit" className="btn btn-primary" disabled={ragLoading}>
                  <Send size={16} />
                </button>
              </form>
            </div>
          )}

          {/* Tab 4: Raw Text */}
          {activeTab === 'raw' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <div className="code-block" style={{ flex: 1, maxHeight: 580, overflowY: 'auto' }}>
                {raw_text || 'No raw text available.'}
              </div>
            </div>
          )}

          {/* Tab 5: Validation & Extraction Metadata */}
          {activeTab === 'validation' && (
            <div style={{ overflowY: 'auto', flex: 1 }}>
              <h3 style={{ fontSize: '0.98rem', fontWeight: 600, marginBottom: 12 }}>
                Extraction Pipeline Provenance
              </h3>
              <div className="card" style={{ padding: 16, marginBottom: 20 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: '0.85rem' }}>
                  <div>
                    <span style={{ color: 'var(--text-dim)' }}>PDF Ingestion Method: </span>
                    <strong style={{ color: 'var(--text-main)' }}>{extraction_metadata?.pdf_method || 'N/A'}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-dim)' }}>OCR Engine: </span>
                    <strong style={{ color: 'var(--text-main)' }}>{extraction_metadata?.ocr_engine || 'None'}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-dim)' }}>LLM Provider: </span>
                    <strong style={{ color: 'var(--text-main)' }}>{extraction_metadata?.llm_provider || 'Heuristic Fallback'}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-dim)' }}>Extraction Mode: </span>
                    <strong style={{ color: 'var(--text-main)' }}>{extraction_metadata?.extraction_mode || 'ocr_only'}</strong>
                  </div>
                </div>
              </div>

              <h3 style={{ fontSize: '0.98rem', fontWeight: 600, marginBottom: 12 }}>
                Audit Validation Details
              </h3>
              <div className="card" style={{ padding: 16 }}>
                <div style={{ marginBottom: 10 }}>
                  Validation Status: <strong className={`badge badge-${validation?.status}`}>{validation?.status}</strong>
                </div>
                {validation?.issues?.length === 0 ? (
                  <div style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.88rem' }}>
                    <CheckCircle2 size={16} /> All structural and numerical checks passed without discrepancies.
                  </div>
                ) : (
                  validation?.issues?.map((iss, idx) => (
                    <div key={idx} style={{ padding: '8px 12px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: 6, marginBottom: 8, fontSize: '0.82rem' }}>
                      <span style={{ color: 'var(--warning)', fontWeight: 600 }}>[{iss.issue_type}]</span> {iss.message}
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
