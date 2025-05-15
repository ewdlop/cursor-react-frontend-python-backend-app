import React, { useState } from 'react';
import { useAnalyzeTextMutation } from '../store/api';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

interface Entity {
  text: string;
  label: string;
}

interface PosTag {
  text: string;
  pos: string;
}

interface AnalysisResult {
  entities: Entity[];
  tokens: string[];
  pos_tags: PosTag[];
}

const TextAnalysis: React.FC = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string>('');
  const [analyzeText, { isLoading }] = useAnalyzeTextMutation();
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setResult(null);

    if (!text.trim()) {
      setError('请输入要分析的文本');
      return;
    }

    try {
      const response = await analyzeText({ text }).unwrap();
      setResult(response.result);
    } catch (err: any) {
      // Handle different types of error responses
      if (typeof err.data === 'string') {
        setError(err.data);
      } else if (err.data?.detail) {
        setError(err.data.detail);
      } else if (err.error) {
        setError(err.error);
      } else {
        setError('分析失败，请稍后重试');
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
        {error && <div className="error">{error}</div>}
        <div className="form-group">
          <label htmlFor="text">输入文本：</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="请输入要分析的文本..."
            required
          />
        </div>
        <button type="submit" disabled={isLoading}>
          {isLoading ? '分析中...' : '分析文本'}
        </button>
      </form>

      {result && (
        <div className="analysis-results">
          <div className="result-section">
            <h3>实体识别</h3>
            <ul>
              {result.entities.map(({ text, label }, index: number) => (
                <li key={`entity-${index}`}>
                  {text} <span className="entity-label">({label})</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="result-section">
            <h3>词性标注</h3>
            <ul>
              {result.pos_tags.map(({ text, pos }, index: number) => (
                <li key={`pos-${index}`}>
                  {text} <span className="pos-tag">({pos})</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="result-section">
            <h3>分词结果</h3>
            <ul>
              {result.tokens.map((token: string, index: number) => (
                <li key={`token-${index}`}>{token}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default TextAnalysis; 