import React, { useState } from 'react';
import { LayoutDashboard, FileUp, Files, Code, Layers } from 'lucide-react';
import DashboardView from './components/DashboardView';
import UploadView from './components/UploadView';
import ExplorerView from './components/ExplorerView';
import DocumentDetailView from './components/DocumentDetailView';
import ApiDocsView from './components/ApiDocsView';

export default function App() {
  const [currentView, setCurrentView] = useState('dashboard'); // dashboard, upload, explorer, detail, apidocs
  const [selectedDocId, setSelectedDocId] = useState(null);

  const handleSelectDocument = (id) => {
    setSelectedDocId(id);
    setCurrentView('detail');
  };

  const handleNavigate = (view) => {
    setCurrentView(view);
    if (view !== 'detail') {
      setSelectedDocId(null);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">
            <Layers size={22} />
          </div>
          <div>
            <div className="brand-title">DocuIntel</div>
            <span className="brand-badge">Universal Platform</span>
          </div>
        </div>

        <ul className="nav-menu">
          <li 
            className={`nav-item ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => handleNavigate('dashboard')}
          >
            <LayoutDashboard size={18} /> Dashboard
          </li>
          <li 
            className={`nav-item ${currentView === 'upload' ? 'active' : ''}`}
            onClick={() => handleNavigate('upload')}
          >
            <FileUp size={18} /> Ingest Document
          </li>
          <li 
            className={`nav-item ${currentView === 'explorer' || currentView === 'detail' ? 'active' : ''}`}
            onClick={() => handleNavigate('explorer')}
          >
            <Files size={18} /> Document Explorer
          </li>
          <li 
            className={`nav-item ${currentView === 'apidocs' ? 'active' : ''}`}
            onClick={() => handleNavigate('apidocs')}
          >
            <Code size={18} /> API Reference
          </li>
        </ul>

        {/* Footer info in sidebar */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border-subtle)', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          <div>DocuIntel Engine v1.0</div>
          <div style={{ marginTop: 2 }}>PyMuPDF • Tesseract • Open-LLM</div>
        </div>
      </aside>

      {/* Main Content View */}
      <main className="main-content">
        {currentView === 'dashboard' && (
          <DashboardView 
            onNavigate={handleNavigate} 
            onSelectDocument={handleSelectDocument} 
          />
        )}

        {currentView === 'upload' && (
          <UploadView 
            onSelectDocument={handleSelectDocument} 
          />
        )}

        {currentView === 'explorer' && (
          <ExplorerView 
            onSelectDocument={handleSelectDocument} 
          />
        )}

        {currentView === 'detail' && selectedDocId && (
          <DocumentDetailView 
            documentId={selectedDocId} 
            onBack={() => handleNavigate('explorer')} 
          />
        )}

        {currentView === 'apidocs' && (
          <ApiDocsView />
        )}
      </main>
    </div>
  );
}
