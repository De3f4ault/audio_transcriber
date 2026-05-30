import { useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { Send, Bot, Trash2, StopCircle } from 'lucide-react';

export default function ChatView() {
  const { messages, setMessages, sendMessage, stop, status } = useChat({ api: '/api/chat' } as any) as any;
  const [input, setInput] = useState('');

  const isLoading = status === 'submitted' || status === 'streaming';

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    sendMessage({ text: input });
    setInput('');
  };
  
  return (
    <div className="card flex-col" style={{ height: 'calc(100vh - 120px)', padding: 0, overflow: 'hidden' }}>
      <div style={{ padding: '24px', borderBottom: '1px solid var(--color-border-subtle)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 className="font-serif">AI Chat</h2>
        <div className="flex gap-2">
          {isLoading && (
            <button className="btn btn-secondary" onClick={() => stop()} title="Stop generating">
              <StopCircle size={18} />
            </button>
          )}
          <button className="btn btn-secondary" onClick={() => setMessages([])} title="Clear Chat">
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {!(messages && messages.length > 0) ? (
          <div className="text-secondary" style={{ margin: 'auto', textAlign: 'center' }}>
            Ask a question about your transcriptions.
          </div>
        ) : (
          (messages || []).map((m: any) => (
            <div key={m.id} className="flex gap-4" style={{
              alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '85%'
            }}>
              {m.role === 'assistant' && (
                <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'var(--color-surface-sand)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <Bot size={18} className="text-brand" />
                </div>
              )}
              <div style={{
                background: m.role === 'user' ? 'var(--color-brand)' : 'var(--color-bg-elevated)',
                color: m.role === 'user' ? '#fff' : 'var(--color-text-primary)',
                padding: '12px 16px',
                borderRadius: '12px',
                border: m.role === 'assistant' ? '1px solid var(--color-border-strong)' : 'none',
                boxShadow: m.role === 'assistant' ? '0 2px 8px rgba(0,0,0,0.02)' : 'none',
                whiteSpace: 'pre-wrap'
              }}>
                {m.content}
              </div>
            </div>
          ))
        )}
      </div>

      <form onSubmit={onSubmit} style={{ padding: '24px', borderTop: '1px solid var(--color-border-subtle)' }} className="flex gap-4">
        <input
          className="input-field"
          value={input || ""}
          placeholder="Ask anything..."
          onChange={(e) => setInput(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" className="btn btn-primary" disabled={!(input || "").trim() || isLoading}>
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
