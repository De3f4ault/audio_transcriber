import { useEffect, useState } from 'react';
import { History as HistoryIcon, Trash2, Search } from 'lucide-react';

export default function HistoryView() {
  const [history, setHistory] = useState<any[]>([]);
  const [search, setSearch] = useState('');
  
  const fetchHistory = async () => {
    try {
      const res = await fetch(`/api/history?limit=50&search=${encodeURIComponent(search)}`);
      const data = await res.json();
      setHistory(data);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [search]);

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this transcription?')) return;
    await fetch(`/api/history/${id}`, { method: 'DELETE' });
    fetchHistory();
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="font-serif flex items-center gap-2"><HistoryIcon /> History</h2>
        <div className="flex items-center gap-2">
          <Search size={18} className="text-secondary" />
          <input 
            className="input-field" 
            placeholder="Search transcripts..." 
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ width: '250px' }}
          />
        </div>
      </div>

      <div className="flex-col gap-4">
        {history.length === 0 ? (
          <div className="card text-center text-secondary">No history found.</div>
        ) : history.map(item => (
          <div key={item.id} className="card flex items-center justify-between">
            <div>
              <h3 className="font-serif" style={{ fontSize: '20px' }}>{item.file_name}</h3>
              <div className="text-sm text-secondary flex gap-4" style={{ marginTop: '8px' }}>
                <span>ID: #{item.id}</span>
                <span>Model: {item.model}</span>
                <span>Duration: {formatDuration(item.duration || 0)}</span>
                <span>Date: {item.created_at?.slice(0, 10)}</span>
              </div>
              {item.text_preview && (
                <p className="text-tertiary text-sm" style={{ marginTop: '8px', fontStyle: 'italic' }}>
                  "{item.text_preview.slice(0, 120)}..."
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <button className="btn btn-secondary" onClick={() => handleDelete(item.id)} title="Delete">
                <Trash2 size={18} className="text-error" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
