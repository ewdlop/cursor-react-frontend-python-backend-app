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

interface Sentiment {
  polarity: number;
  subjectivity: number;
}

interface Keyword {
  word: string;
  weight: number;
}

interface Dependency {
  text: string;
  dep: string;
  head: string;
}

interface AnalysisResult {
  entities: Entity[];
  tokens: string[];
  pos_tags: PosTag[];
  sentiment?: Sentiment;
  keywords?: Keyword[];
  word_frequency?: Record<string, number>;
  dependencies?: Dependency[];
}

const TextAnalysis: React.FC = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string>('');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(['basic']);
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
      console.log('Sending analysis request:', { text, features: selectedFeatures });
      const response = await analyzeText({ 
        text,
        features: selectedFeatures
      }).unwrap();
      
      console.log('Received response:', response);
      
      // Transform the API response to match the expected types
      const transformedResult: AnalysisResult = {
        entities: response.result.entities.map(([text, label]) => ({ text, label })),
        tokens: response.result.tokens,
        pos_tags: response.result.pos_tags.map(([text, pos]) => ({ text, pos })),
        sentiment: response.result.sentiment,
        keywords: response.result.keywords?.map(([word, weight]) => ({ word, weight })),
        word_frequency: response.result.word_frequency,
        dependencies: response.result.dependencies?.map(([text, dep, head]) => ({ text, dep, head }))
      };
      
      console.log('Transformed result:', transformedResult);
      setResult(transformedResult);
    } catch (err: any) {
      console.error('Analysis error:', err);
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

  const handleFeatureToggle = (feature: string) => {
    setSelectedFeatures(prev => 
      prev.includes(feature)
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    );
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
            disabled={isLoading}
          />
        </div>
        
        <div className="feature-selection">
          <h3>选择分析特征：</h3>
          <div className="feature-buttons">
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('basic') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('basic')}
              disabled={isLoading}
            >
              基础分析
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('sentiment') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('sentiment')}
              disabled={isLoading}
            >
              情感分析
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('keywords') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('keywords')}
              disabled={isLoading}
            >
              关键词提取
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('word_freq') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('word_freq')}
              disabled={isLoading}
            >
              词频统计
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('dependencies') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('dependencies')}
              disabled={isLoading}
            >
              依存句法
            </button>
          </div>
        </div>

        <button type="submit" disabled={isLoading || !text.trim()}>
          {isLoading ? '分析中...' : '分析文本'}
        </button>
      </form>

      {isLoading && (
        <div className="loading-indicator">
          正在分析文本，请稍候...
        </div>
      )}

      {result && !isLoading && (
        <div className="analysis-results">
          {selectedFeatures.includes('basic') && (
            <>
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
            </>
          )}

          {selectedFeatures.includes('sentiment') && result.sentiment && (
            <div className="result-section">
              <h3>情感分析</h3>
              <div className="sentiment-info">
                <p>情感极性: {result.sentiment.polarity.toFixed(2)}</p>
                <p>主观程度: {result.sentiment.subjectivity.toFixed(2)}</p>
              </div>
            </div>
          )}

          {selectedFeatures.includes('keywords') && result.keywords && (
            <div className="result-section">
              <h3>关键词提取</h3>
              <ul>
                {result.keywords.map(({ word, weight }, index: number) => (
                  <li key={`keyword-${index}`}>
                    {word} <span className="weight-tag">({weight.toFixed(2)})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {selectedFeatures.includes('word_freq') && result.word_frequency && (
            <div className="result-section">
              <h3>词频统计</h3>
              <ul>
                {Object.entries(result.word_frequency).map(([word, freq], index: number) => (
                  <li key={`freq-${index}`}>
                    {word} <span className="freq-tag">({freq})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {selectedFeatures.includes('dependencies') && result.dependencies && (
            <div className="result-section">
              <h3>依存句法分析</h3>
              <ul>
                {result.dependencies.map(({ text, dep, head }, index: number) => (
                  <li key={`dep-${index}`}>
                    {text} <span className="dep-tag">({dep})</span> → {head}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TextAnalysis; 