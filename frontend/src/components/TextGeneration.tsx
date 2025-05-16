import React, { useState } from 'react';
import { useGenerateTextMutation, useGetTextGenerationHistoryQuery } from '../store/api';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import '../styles/components.css';

interface TextGenerationRequest {
  prompt: string;
  max_length: number;
  temperature: number;
  top_p: number;
  num_return_sequences: number;
}

interface TextGenerationResult {
  id: string;
  prompt: string;
  generated_text: string;
  timestamp: string;
  username: string;
  settings: {
    max_length: number;
    temperature: number;
    top_p: number;
    num_return_sequences: number;
  };
}

const TextGeneration: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [settings, setSettings] = useState({
    max_length: 100,
    temperature: 0.7,
    top_p: 0.9,
    num_return_sequences: 1
  });
  const [error, setError] = useState<string>('');
  const [showJson, setShowJson] = useState(false);
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);
  
  const [generateText, { isLoading }] = useGenerateTextMutation();
  const { data: history } = useGetTextGenerationHistoryQuery();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError('请输入提示文本');
      return;
    }

    setError('');

    try {
      await generateText({
        prompt,
        ...settings
      }).unwrap();
      
      setPrompt('');
    } catch (err: any) {
      console.error('Generation error:', err);
      setError(err.data?.detail || '生成失败，请重试');
    }
  };

  const handleSettingChange = (setting: keyof typeof settings, value: number) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  if (!isAuthenticated) {
    return <div>请先登录</div>;
  }

  return (
    <div className="text-analysis-container">
      <h2>AI 文本生成</h2>
      <form onSubmit={handleSubmit} className="analysis-form">
        {error && <div className="error">{error}</div>}
        
        <div className="form-group">
          <label htmlFor="prompt">提示文本：</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="输入提示文本..."
            disabled={isLoading}
            rows={4}
            className="prompt-input"
          />
        </div>

        <div className="generation-settings">
          <h3>生成设置：</h3>
          <div className="settings-grid">
            <div className="setting-item">
              <label htmlFor="max_length">最大长度：</label>
              <input
                type="range"
                id="max_length"
                min="50"
                max="500"
                step="10"
                value={settings.max_length}
                onChange={(e) => handleSettingChange('max_length', Number(e.target.value))}
                disabled={isLoading}
              />
              <span>{settings.max_length}</span>
            </div>

            <div className="setting-item">
              <label htmlFor="temperature">温度：</label>
              <input
                type="range"
                id="temperature"
                min="0.1"
                max="1.0"
                step="0.1"
                value={settings.temperature}
                onChange={(e) => handleSettingChange('temperature', Number(e.target.value))}
                disabled={isLoading}
              />
              <span>{settings.temperature}</span>
            </div>

            <div className="setting-item">
              <label htmlFor="top_p">Top P：</label>
              <input
                type="range"
                id="top_p"
                min="0.1"
                max="1.0"
                step="0.1"
                value={settings.top_p}
                onChange={(e) => handleSettingChange('top_p', Number(e.target.value))}
                disabled={isLoading}
              />
              <span>{settings.top_p}</span>
            </div>

            <div className="setting-item">
              <label htmlFor="num_return_sequences">生成数量：</label>
              <input
                type="range"
                id="num_return_sequences"
                min="1"
                max="5"
                step="1"
                value={settings.num_return_sequences}
                onChange={(e) => handleSettingChange('num_return_sequences', Number(e.target.value))}
                disabled={isLoading}
              />
              <span>{settings.num_return_sequences}</span>
            </div>
          </div>
        </div>

        <button 
          type="submit" 
          disabled={isLoading}
          className="submit-button"
        >
          {isLoading ? '生成中...' : '开始生成'}
        </button>
      </form>

      {isLoading && (
        <div className="loading-indicator">
          正在生成文本，请稍候...
        </div>
      )}

      {history && history.length > 0 && (
        <div className="history-section">
          <h3>生成历史</h3>
          <div className="view-toggle">
            <button
              type="button"
              className={`view-button ${!showJson ? 'active' : ''}`}
              onClick={() => setShowJson(false)}
            >
              可视化视图
            </button>
            <button
              type="button"
              className={`view-button ${showJson ? 'active' : ''}`}
              onClick={() => setShowJson(true)}
            >
              JSON视图
            </button>
          </div>
          <div className="history-list">
            {history.map((item) => (
              <div key={item.id} className="history-item">
                <div className="history-header">
                  <p className="history-text">{item.prompt}</p>
                  <p className="history-time">{new Date(item.timestamp).toLocaleString()}</p>
                </div>
                {showJson ? (
                  <pre className="json-view">
                    {JSON.stringify(item, null, 2)}
                  </pre>
                ) : (
                  <div className="result-content">
                    <div className="generation-info">
                      <p><strong>生成文本：</strong>{item.generated_text}</p>
                      <p><strong>最大长度：</strong>{item.settings.max_length}</p>
                      <p><strong>温度：</strong>{item.settings.temperature}</p>
                      <p><strong>Top P：</strong>{item.settings.top_p}</p>
                      <p><strong>生成数量：</strong>{item.settings.num_return_sequences}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TextGeneration; 