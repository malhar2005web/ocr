import React, { useState } from 'react';
import { LayoutDashboard, FileUp, Files, Code } from 'lucide-react';
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
          <img 
            src="/pcs_logo.png" 
            alt="PCS Logo" 
            className="brand-logo-img" 
          />
          <div>
            <div className="brand-title">PCS DocuIntel</div>
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
