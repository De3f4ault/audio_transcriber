import { useState, useEffect } from 'react';
import { Play, XCircle, FileAudio } from 'lucide-react';

export default function TranscribeView() {
  const [file, setFile] = useState<File | null>(null);
  const [engine, setEngine] = useState('whisper');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [progress, setProgress] = useState(0);
  const [segments, setSegments] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const startTranscription = async () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('engine', engine);
    
    try {
      setStatus('starting');
      setSegments([]);
      setProgress(0);
      setError(null);
      
      const res = await fetch('/api/transcribe/', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      setJobId(data.job_id);
    } catch (err: any) {
      setError(err.message);
      setStatus('error');
    }
  };

  useEffect(() => {
    if (!jobId) return;
    
    setStatus('processing');
    const eventSource = new EventSource(`/api/transcribe/stream/${jobId}`);
    
    eventSource.addEventListener('segment', (e) => {
      const seg = JSON.parse(e.data);
      setSegments(prev => [...prev, seg]);
    });
    
    eventSource.addEventListener('progress', (e) => {
      const data = JSON.parse(e.data);
      setProgress(data.progress);
    });
    
    eventSource.addEventListener('status', (e) => {
      const data = JSON.parse(e.data);
      setStatus(data.status);
      if (data.status === 'error') setError(data.error);
      eventSource.close();
    });
    
    eventSource.onerror = () => {
      eventSource.close();
    };
    
    return () => eventSource.close();
  }, [jobId]);

  const cancelJob = async () => {
    if (jobId) {
      await fetch(`/api/transcribe/${jobId}`, { method: 'DELETE' });
    }
  };

  return (
    <div className="flex-col gap-6">
      <div className="card">
        <h2 className="font-serif text-brand flex items-center gap-2">
          <FileAudio /> New Transcription
        </h2>
        <div className="flex gap-4" style={{ marginTop: '24px' }}>
          <div style={{ flex: 1 }}>
            <label className="text-sm text-secondary block" style={{ marginBottom: '8px' }}>Audio File</label>
            <input 
              type="file" 
              accept="audio/*,video/*"
              className="input-field" 
              onChange={e => setFile(e.target.files?.[0] || null)} 
              disabled={status === 'processing'}
            />
          </div>
          <div style={{ width: '200px' }}>
            <label className="text-sm text-secondary block" style={{ marginBottom: '8px' }}>Engine</label>
            <select 
              className="input-field" 
              value={engine} 
              onChange={e => setEngine(e.target.value)}
              disabled={status === 'processing'}
            >
              <option value="whisper">Whisper</option>
              <option value="gemini">Gemini</option>
            </select>
          </div>
        </div>
        <div style={{ marginTop: '24px' }}>
          {status === 'processing' ? (
            <button className="btn btn-secondary" onClick={cancelJob} style={{ color: 'var(--color-error)' }}>
              <XCircle size={18} /> Cancel
            </button>
          ) : (
            <button className="btn btn-primary" onClick={startTranscription} disabled={!file || status === 'starting'}>
              <Play size={18} /> Transcribe
            </button>
          )}
        </div>
      </div>

      {(status === 'processing' || status === 'completed') && (
        <div className="card">
          <div className="flex items-center justify-between" style={{ marginBottom: '16px' }}>
            <h3 className="font-serif">Live Output</h3>
            <span className="text-sm text-secondary">{progress.toFixed(1)}%</span>
          </div>
          
          <div style={{
            background: 'var(--color-surface-sand)',
            borderRadius: '4px',
            height: '4px',
            overflow: 'hidden',
            marginBottom: '24px'
          }}>
            <div style={{
              background: 'var(--color-brand)',
              width: `${progress}%`,
              height: '100%',
              transition: 'width 0.3s ease'
            }} />
          </div>

          <div className="text-primary" style={{ fontSize: '18px', lineHeight: 1.8 }}>
            {segments.map((seg, i) => (
              <span key={i} title={`[${seg.start.toFixed(1)}s]`} style={{ marginRight: '6px' }}>
                {seg.text}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {error && (
        <div className="card" style={{ borderLeft: '4px solid var(--color-error)' }}>
          <p className="text-error">{error}</p>
        </div>
      )}
    </div>
  );
}
