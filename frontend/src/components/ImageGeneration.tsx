import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { useGenerateImageMutation, useGetGenerationHistoryQuery } from '../store/api';

const ImageGeneration: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [settings, setSettings] = useState({
    numSteps: 50,
    guidanceScale: 7.5,
    width: 512,
    height: 512,
  });
  const [error, setError] = useState('');
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);
  
  const [generateImage, { isLoading }] = useGenerateImageMutation();
  const { data: history } = useGetGenerationHistoryQuery();

  const handleSettingChange = (setting: string, value: number) => {
    setSettings(prev => ({
      ...prev,
      [setting]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) {
      setError('请输入提示词');
      return;
    }

    setError('');

    try {
      const result = await generateImage({
        prompt,
        negative_prompt: negativePrompt,
        ...settings
      }).unwrap();
      
      console.log('Generation result:', result);
      setError('');
      
    } catch (err: any) {
      console.error('Generation error:', err);
      setError(err.data?.detail || '生成失败，请重试');
    }
  };

  if (!isAuthenticated) {
    return <div>请先登录</div>;
  }

  return (
    <div className="text-analysis-container">
      <h2>AI 图像生成</h2>
      <form onSubmit={handleSubmit} className="analysis-form">
        {error && <div className="error">{error}</div>}
        
        <div className="form-group">
          <label htmlFor="prompt">提示词：</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="描述你想要生成的图像..."
            disabled={isLoading}
            className="prompt-input"
          />
        </div>

        <div className="form-group">
          <label htmlFor="negative-prompt">负面提示词：</label>
          <textarea
            id="negative-prompt"
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder="描述你不想在图像中出现的元素..."
            disabled={isLoading}
            className="prompt-input"
          />
        </div>

        <div className="generation-settings">
          <h3>生成设置：</h3>
          <div className="settings-grid">
            <div className="setting-item">
              <label>生成步数：</label>
              <input
                type="range"
                min="20"
                max="100"
                step="1"
                value={settings.numSteps}
                onChange={(e) => handleSettingChange('numSteps', parseInt(e.target.value))}
                disabled={isLoading}
              />
              <span>{settings.numSteps}</span>
            </div>
            
            <div className="setting-item">
              <label>引导系数：</label>
              <input
                type="range"
                min="1"
                max="20"
                step="0.1"
                value={settings.guidanceScale}
                onChange={(e) => handleSettingChange('guidanceScale', parseFloat(e.target.value))}
                disabled={isLoading}
              />
              <span>{settings.guidanceScale.toFixed(1)}</span>
            </div>
            
            <div className="setting-item">
              <label>宽度：</label>
              <select
                value={settings.width}
                onChange={(e) => handleSettingChange('width', parseInt(e.target.value))}
                disabled={isLoading}
              >
                <option value="256">256</option>
                <option value="384">384</option>
                <option value="512">512</option>
                <option value="768">768</option>
              </select>
            </div>
            
            <div className="setting-item">
              <label>高度：</label>
              <select
                value={settings.height}
                onChange={(e) => handleSettingChange('height', parseInt(e.target.value))}
                disabled={isLoading}
              >
                <option value="256">256</option>
                <option value="384">384</option>
                <option value="512">512</option>
                <option value="768">768</option>
              </select>
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
          正在生成图像，请稍候...
        </div>
      )}

      {history && history.length > 0 && (
        <div className="history-section">
          <h3>生成历史</h3>
          <div className="history-grid">
            {history.map((item) => (
              <div key={item.id} className="history-item">
                <img src={item.image_url} alt="Generated" className="history-image" />
                <div className="generation-info">
                  <p><strong>提示词：</strong>{item.prompt}</p>
                  {item.negative_prompt && (
                    <p><strong>负面提示词：</strong>{item.negative_prompt}</p>
                  )}
                  <p><strong>生成时间：</strong>{new Date(item.timestamp).toLocaleString()}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageGeneration; 