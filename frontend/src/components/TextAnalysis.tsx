import React, { useState } from 'react';
import { useAnalyzeTextMutation } from '../store/api';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

const TextAnalysis: React.FC = () => {
  const [text, setText] = useState('');
  const [analyzeText, { data, isLoading, error }] = useAnalyzeTextMutation();
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      try {
        await analyzeText({ text }).unwrap();
      } catch (err) {
        console.error('分析失败:', err);
      }
    }
  };

  if (!isAuthenticated) {
    return <div>请先登录</div>;
  }

  return (
    <div className="text-analysis-container">
      <h2>文本分析</h2>
      <form onSubmit={handleSubmit} className="analysis-form">
        <div className="form-group">
          <label htmlFor="text">输入文本：</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            required
            rows={5}
          />
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? '分析中...' : '分析文本'}
        </button>
      </form>

      {error && <div className="error">分析失败，请重试</div>}

      {data && (
        <div className="analysis-results">
          <h3>分析结果：</h3>
          <div className="result-section">
            <h4>实体：</h4>
            <ul>
              {data.result.entities.map(([text, label], index) => (
                <li key={index}>
                  {text} ({label})
                </li>
              ))}
            </ul>
          </div>
          <div className="result-section">
            <h4>词性标注：</h4>
            <ul>
              {data.result.pos_tags.map(([text, pos], index) => (
                <li key={index}>
                  {text} ({pos})
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default TextAnalysis; 