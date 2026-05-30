import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { FileAudio, MessageSquare, History, Settings } from 'lucide-react'

import TranscribeView from './views/TranscribeView'
import ChatView from './views/ChatView'
import HistoryView from './views/HistoryView'
import SettingsView from './views/SettingsView'

function App() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Transcribe', icon: <FileAudio size={18} /> },
    { path: '/chat', label: 'Chat', icon: <MessageSquare size={18} /> },
    { path: '/history', label: 'History', icon: <History size={18} /> },
    { path: '/settings', label: 'Settings', icon: <Settings size={18} /> },
  ];

  return (
    <div className="flex-col" style={{ minHeight: '100vh' }}>
      {/* Navigation Bar */}
      <nav style={{ 
        borderBottom: '1px solid var(--color-border-subtle)',
        backgroundColor: 'var(--color-bg-primary)',
        position: 'sticky',
        top: 0,
        zIndex: 10
      }}>
        <div className="container flex items-center justify-between" style={{ height: '64px' }}>
          <div className="flex items-center gap-2">
            <div style={{
              width: '32px', height: '32px', borderRadius: '8px', 
              background: 'var(--color-brand)', display: 'flex', 
              alignItems: 'center', justifyContent: 'center', color: '#fff'
            }}>
              <FileAudio size={18} />
            </div>
            <span className="font-serif" style={{ fontSize: '21px', fontWeight: 500, color: 'var(--color-text-primary)' }}>
              AudioBench
            </span>
          </div>
          
          <div className="flex gap-2">
            {navItems.map((item) => (
              <Link 
                key={item.path} 
                to={item.path}
                className="btn btn-secondary"
                style={{
                  background: location.pathname === item.path ? 'var(--color-surface-sand)' : 'transparent',
                  boxShadow: location.pathname === item.path ? 'var(--color-surface-sand) 0 0 0 0, var(--color-ring-default) 0 0 0 1px' : 'none',
                }}
              >
                {item.icon}
                {item.label}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="container" style={{ flex: 1, padding: '40px 24px' }}>
        <Routes>
          <Route path="/" element={<TranscribeView />} />
          <Route path="/chat" element={<ChatView />} />
          <Route path="/history" element={<HistoryView />} />
          <Route path="/settings" element={<SettingsView />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
