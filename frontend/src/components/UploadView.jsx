import React, { useState, useRef } from 'react';
import { Upload, FileUp, CheckCircle2, AlertCircle, Loader2, ArrowRight, Eye, Trash2 } from 'lucide-react';
import { api } from '../services/api';

export default function UploadView({ onSelectDocument }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [isUploading, setIsUploading] = useState(false);
  const inputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
    }
  };

  const addFiles = (newFiles) => {
    const valid = newFiles.filter(f => {
      const ext = f.name.split('.').pop().toLowerCase();
      return ['pdf', 'jpg', 'jpeg', 'png'].includes(ext);
    });
    setSelectedFiles(prev => [...prev, ...valid]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const processAllFiles = async () => {
    if (selectedFiles.length === 0) return;
    setIsUploading(true);

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      const fileKey = `${file.name}-${i}`;

      // Update state to processing
      setUploadProgress(prev => ({
        ...prev,
        [fileKey]: { status: 'uploading', message: 'Extracting text and structure...' }
      }));

      try {
        const response = await api.uploadDocument(file, false);
        setUploadProgress(prev => ({
          ...prev,
          [fileKey]: {
            status: 'completed',
            document_id: response.document_id,
            validation_status: response.status,
            message: 'Extraction completed successfully!'
          }
        }));
      } catch (err) {
        setUploadProgress(prev => ({
          ...prev,
          [fileKey]: {
            status: 'error',
            message: err.message || 'Extraction failed'
          }
        }));
      }
    }
    setIsUploading(false);
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: 900, margin: '0 auto' }}>
      <div className="page-header" style={{ textAlign: 'center' }}>
        <h1 className="page-title">Ingest & Extract Documents</h1>
        <p className="page-desc">
          Upload any business document, invoice, challan, contract, or scanned image to extract normalized JSON.
        </p>
      </div>

      {/* Drag & Drop Upload Zone */}
      <div 
        className={`upload-dropzone ${dragActive ? 'drag-active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input 
          ref={inputRef}
          type="file" 
          multiple 
          accept=".pdf,.png,.jpg,.jpeg"
          style={{ display: 'none' }}
          onChange={handleChange}
        />
        <div style={{ width: 64, height: 64, margin: '0 auto 16px', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--primary)' }}>
          <FileUp size={32} />
        </div>
        <h3 style={{ fontSize: '1.2rem', fontWeight: 600, color: '#fff', marginBottom: 8 }}>
          Drag & Drop PDF or Image files here
        </h3>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: 18 }}>
          Supports Native PDF, Scanned PDF, JPG, PNG (up to 50MB per file)
        </p>
        <button type="button" className="btn btn-primary" onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>
          <Upload size={16} /> Browse From Computer
        </button>
      </div>

      {/* Selected File Queue */}
      {selectedFiles.length > 0 && (
        <div className="card" style={{ marginTop: 28 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600 }}>
              Files Ready for Processing ({selectedFiles.length})
            </h3>
            {!isUploading && (
              <button className="btn btn-primary" onClick={processAllFiles}>
                Start Extraction ({selectedFiles.length}) <ArrowRight size={16} />
              </button>
            )}
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {selectedFiles.map((file, idx) => {
              const fileKey = `${file.name}-${idx}`;
              const progress = uploadProgress[fileKey];

              return (
                <div 
                  key={fileKey}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '14px 18px',
                    background: 'rgba(15, 23, 42, 0.6)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 'var(--radius-md)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                    <div style={{ padding: 8, borderRadius: 6, background: 'rgba(255, 255, 255, 0.05)', color: 'var(--text-muted)' }}>
                      {file.name.endsWith('.pdf') ? 'PDF' : 'IMG'}
                    </div>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.92rem' }}>{file.name}</div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                        {(file.size / 1024).toFixed(1)} KB • {file.type || 'application/octet-stream'}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                    {!progress && !isUploading && (
                      <button 
                        className="btn btn-secondary" 
                        style={{ padding: 6, border: 'none' }}
                        onClick={() => removeFile(idx)}
                      >
                        <Trash2 size={16} color="var(--text-dim)" />
                      </button>
                    )}

                    {progress?.status === 'uploading' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--primary)', fontSize: '0.85rem' }}>
                        <Loader2 size={16} className="spin" /> Processing...
                      </div>
                    )}

                    {progress?.status === 'completed' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span className={`badge badge-${progress.validation_status}`}>
                          <CheckCircle2 size={13} /> {progress.validation_status}
                        </span>
                        <button 
                          className="btn btn-primary"
                          style={{ padding: '6px 12px', fontSize: '0.82rem' }}
                          onClick={() => onSelectDocument(progress.document_id)}
                        >
                          <Eye size={14} /> View JSON
                        </button>
                      </div>
                    )}

                    {progress?.status === 'error' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: 'var(--error)', fontSize: '0.85rem' }}>
                        <AlertCircle size={16} /> {progress.message}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
