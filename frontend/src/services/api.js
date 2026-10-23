const API_BASE = '/api/v1';

export const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async uploadDocument(file, asyncMode = false) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('async_mode', asyncMode ? 'true' : 'false');

    const res = await fetch(`${API_BASE}/documents/extract`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || 'Document upload failed');
    }
    return res.json();
  },

  async getDocument(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || 'Failed to fetch document');
    }
    return res.json();
  },

  async getDocumentStatus(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}/status`);
    if (!res.ok) throw new Error('Failed to fetch status');
    return res.json();
  },

  async listDocuments(params = {}) {
    const query = new URLSearchParams();
    if (params.documentType) query.append('documentType', params.documentType);
    if (params.filename) query.append('filename', params.filename);
    if (params.status) query.append('status', params.status);
    if (params.fromDate) query.append('fromDate', params.fromDate);
    if (params.toDate) query.append('toDate', params.toDate);
    if (params.page) query.append('page', params.page);
    if (params.limit) query.append('limit', params.limit);
    if (params.sortBy) query.append('sortBy', params.sortBy);
    if (params.sortOrder) query.append('sortOrder', params.sortOrder);
    if (params.json_filter) query.append('json_filter', params.json_filter);

    const res = await fetch(`${API_BASE}/documents?${query.toString()}`);
    if (!res.ok) throw new Error('Failed to fetch document list');
    return res.json();
  },

  async getRawText(docId) {
    const res = await fetch(`${API_BASE}/documents/${docId}/raw`);
    if (!res.ok) throw new Error('Failed to fetch raw text');
    return res.json();
  },

  getDownloadUrl(docId) {
    return `${API_BASE}/documents/${docId}/download`;
  },

  async queryDocumentRAG(docId, query) {
    const res = await fetch(`${API_BASE}/documents/${docId}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.message || 'RAG query failed');
    }
    return res.json();
  }
};
