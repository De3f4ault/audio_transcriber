import { useEffect, useState } from 'react';
import { Settings as SettingsIcon, Save } from 'lucide-react';

export default function SettingsView() {
  const [settings, setSettings] = useState<any>({});
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');

  useEffect(() => {
    fetch('/api/settings/')
      .then(res => res.json())
      .then(data => setSettings(data));
  }, []);

  const handleChange = (key: string, value: any) => {
    setSettings((prev: any) => ({ ...prev, [key]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch('/api/settings/', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
      setMessage('Settings saved successfully!');
      setTimeout(() => setMessage(''), 3000);
    } catch (e) {
      setMessage('Error saving settings.');
    }
    setSaving(false);
  };

  return (
    <div className="card flex-col gap-6">
      <h2 className="font-serif flex items-center gap-2"><SettingsIcon /> Configuration</h2>
      
      <div className="flex-col gap-4" style={{ maxWidth: '600px' }}>
        <div>
          <label className="text-sm text-secondary block" style={{ marginBottom: '8px' }}>Ollama Base URL</label>
          <input 
            className="input-field" 
            value={settings.ollama_base_url || ''} 
            onChange={e => handleChange('ollama_base_url', e.target.value)} 
          />
        </div>
        <div>
          <label className="text-sm text-secondary block" style={{ marginBottom: '8px' }}>Ollama Model</label>
          <input 
            className="input-field" 
            value={settings.ollama_model || ''} 
            onChange={e => handleChange('ollama_model', e.target.value)} 
          />
        </div>
        <div>
          <label className="text-sm text-secondary block" style={{ marginBottom: '8px' }}>Gemini API Key</label>
          <input 
            type="password"
            className="input-field" 
            value={settings.gemini_api_key || ''} 
            onChange={e => handleChange('gemini_api_key', e.target.value)} 
            placeholder="AIza..."
          />
        </div>
        <div>
          <label className="text-sm text-secondary block" style={{ marginBottom: '8px' }}>Default Output Format</label>
          <select 
            className="input-field"
            value={settings.output_format || 'txt'}
            onChange={e => handleChange('output_format', e.target.value)}
          >
            <option value="txt">Text (.txt)</option>
            <option value="srt">Subtitles (.srt)</option>
            <option value="vtt">Subtitles (.vtt)</option>
            <option value="json">JSON (.json)</option>
          </select>
        </div>
        
        <div style={{ marginTop: '16px' }} className="flex items-center gap-4">
          <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
            <Save size={18} /> {saving ? 'Saving...' : 'Save Settings'}
          </button>
          {message && <span className={message.includes('Error') ? 'text-error text-sm' : 'text-brand text-sm'}>{message}</span>}
        </div>
      </div>
    </div>
  );
}
