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

interface TextStats {
  sentence_count: number;
  word_count: number;
  unique_words: number;
  avg_sentence_length: number;
}

interface AnalysisResult {
  id: string;
  text: string;
  result: {
    entities?: [string, string][];
    tokens?: string[];
    pos_tags?: [string, string][];
    sentiment?: {
      polarity: number;
      subjectivity: number;
    };
    keywords?: [string, number][];
    word_frequency?: Record<string, number>;
    dependencies?: [string, string, string][];
    // NLTK features
    sentences?: string[];
    lemmatized?: string[];
    nltk_pos_tags?: [string, string][];
    filtered_words?: string[];
    bigrams?: [string, string][];
    trigrams?: [string, string, string][];
    word_frequency_nltk?: Record<string, number>;
    text_stats?: TextStats;
    summary?: string[];
    language?: string;
    similarity?: {
      cosine_similarity: number;
      edit_distance: number;
    };
    readability?: {
      avg_sentence_length: number;
      unique_word_ratio: number;
      sentence_count: number;
      word_count: number;
    };
    entity_relations?: Array<{
      entity1: string;
      entity2: string;
      relation: string;
    }>;
    text_classification?: {
      category: string;
      confidence: number;
    };
  };
  timestamp: string;
  username: string;
}

interface TextAnalysisRequest {
  text: string;
  features: string[];
  compare_text?: string;
}

const TextAnalysis: React.FC = () => {
  const [text, setText] = useState('');
  const [compareText, setCompareText] = useState('');
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
      const response = await analyzeText({
        text,
        features: selectedFeatures,
        compare_text: compareText
      }).unwrap();
      setResult(response);
    } catch (err: any) {
      if (err.data?.detail) {
        setError(err.data.detail);
      } else {
        setError('分析失败，请重试');
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

        {selectedFeatures.includes('similarity') && (
          <div className="form-group">
            <label htmlFor="compare-text">比较文本：</label>
            <textarea
              id="compare-text"
              value={compareText}
              onChange={(e) => setCompareText(e.target.value)}
              placeholder="请输入要比较的文本..."
              disabled={isLoading}
            />
          </div>
        )}
        
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
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('nltk') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('nltk')}
              disabled={isLoading}
            >
              NLTK分析
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('summarization') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('summarization')}
              disabled={isLoading}
            >
              文本摘要
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('language') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('language')}
              disabled={isLoading}
            >
              语言检测
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('similarity') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('similarity')}
              disabled={isLoading}
            >
              文本相似度
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('readability') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('readability')}
              disabled={isLoading}
            >
              可读性分析
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('entity_relations') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('entity_relations')}
              disabled={isLoading}
            >
              实体关系
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('text_classification') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('text_classification')}
              disabled={isLoading}
            >
              文本分类
            </button>
          </div>
        </div>

        <button type="submit" disabled={isLoading || !text.trim()} className="submit-button">
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
          <h3>分析结果：</h3>
          
          {result.result.entities && (
            <div className="result-section">
              <h4>实体识别：</h4>
              <ul>
                {result.result.entities.map(([text, label], index) => (
                  <li key={index}>{text} ({label})</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.sentiment && (
            <div className="result-section">
              <h4>情感分析：</h4>
              <p>情感极性：{result.result.sentiment.polarity.toFixed(2)}</p>
              <p>主观性：{result.result.sentiment.subjectivity.toFixed(2)}</p>
            </div>
          )}

          {result.result.keywords && (
            <div className="result-section">
              <h4>关键词：</h4>
              <ul>
                {result.result.keywords.map(([word, weight], index) => (
                  <li key={index}>{word} ({weight.toFixed(2)})</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.word_frequency && (
            <div className="result-section">
              <h4>词频统计：</h4>
              <ul>
                {Object.entries(result.result.word_frequency).map(([word, freq], index) => (
                  <li key={index}>{word}: {freq}</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.dependencies && (
            <div className="result-section">
              <h4>依存句法分析：</h4>
              <ul>
                {result.result.dependencies.map(([text, dep, head], index) => (
                  <li key={index}>{text} {'--'}{dep}{'-->'} {head}</li>
                ))}
              </ul>
            </div>
          )}

          {/* NLTK Results */}
          {result.result.text_stats && (
            <div className="result-section">
              <h4>文本统计：</h4>
              <ul>
                <li>句子数量：{result.result.text_stats.sentence_count}</li>
                <li>单词数量：{result.result.text_stats.word_count}</li>
                <li>唯一单词数：{result.result.text_stats.unique_words}</li>
                <li>平均句子长度：{result.result.text_stats.avg_sentence_length.toFixed(2)}</li>
              </ul>
            </div>
          )}

          {result.result.sentences && (
            <div className="result-section">
              <h4>句子分割：</h4>
              <ol>
                {result.result.sentences.map((sentence, index) => (
                  <li key={index}>{sentence}</li>
                ))}
              </ol>
            </div>
          )}

          {result.result.lemmatized && (
            <div className="result-section">
              <h4>词形还原：</h4>
              <p>{result.result.lemmatized.join(' ')}</p>
            </div>
          )}

          {result.result.nltk_pos_tags && (
            <div className="result-section">
              <h4>词性标注：</h4>
              <ul>
                {result.result.nltk_pos_tags.map(([text, pos], index) => (
                  <li key={index}>{text} ({pos})</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.filtered_words && (
            <div className="result-section">
              <h4>停用词过滤：</h4>
              <p>{result.result.filtered_words.join(' ')}</p>
            </div>
          )}

          {result.result.bigrams && (
            <div className="result-section">
              <h4>二元语法：</h4>
              <ul>
                {result.result.bigrams.map(([word1, word2], index) => (
                  <li key={index}>{word1} {word2}</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.trigrams && (
            <div className="result-section">
              <h4>三元语法：</h4>
              <ul>
                {result.result.trigrams.map(([word1, word2, word3], index) => (
                  <li key={index}>{word1} {word2} {word3}</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.word_frequency_nltk && (
            <div className="result-section">
              <h4>NLTK词频统计：</h4>
              <ul>
                {Object.entries(result.result.word_frequency_nltk).map(([word, freq], index) => (
                  <li key={index}>{word}: {freq}</li>
                ))}
              </ul>
            </div>
          )}

          {result.result.summary && (
            <div className="result-section">
              <h4>文本摘要：</h4>
              <ol>
                {result.result.summary.map((sentence, index) => (
                  <li key={index}>{sentence}</li>
                ))}
              </ol>
            </div>
          )}

          {result.result.language && (
            <div className="result-section">
              <h4>语言检测：</h4>
              <p>检测到的语言：{result.result.language}</p>
            </div>
          )}

          {result.result.similarity && (
            <div className="result-section">
              <h4>文本相似度：</h4>
              <p>余弦相似度：{(result.result.similarity.cosine_similarity * 100).toFixed(2)}%</p>
              <p>编辑距离：{result.result.similarity.edit_distance}</p>
            </div>
          )}

          {result.result.readability && (
            <div className="result-section">
              <h4>可读性分析：</h4>
              <ul>
                <li>平均句子长度：{result.result.readability.avg_sentence_length.toFixed(2)}</li>
                <li>唯一词比例：{(result.result.readability.unique_word_ratio * 100).toFixed(2)}%</li>
                <li>句子数量：{result.result.readability.sentence_count}</li>
                <li>词数：{result.result.readability.word_count}</li>
              </ul>
            </div>
          )}

          {result.result.entity_relations && (
            <div className="result-section">
              <h4>实体关系：</h4>
              <ul>
                {result.result.entity_relations.map((relation, index) => (
                  <li key={index}>
                    {relation.entity1} {'--'}{relation.relation}{'-->'} {relation.entity2}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.result.text_classification && (
            <div className="result-section">
              <h4>文本分类：</h4>
              <p>分类：{result.result.text_classification.category}</p>
              <p>置信度：{(result.result.text_classification.confidence * 100).toFixed(2)}%</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TextAnalysis; 