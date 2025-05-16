import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { useProcessImageMutation, useGetImageHistoryQuery } from '../store/api';

const ImageProcessing: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(['basic']);
  const [error, setError] = useState<string>('');
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);
  
  const [processImage, { isLoading }] = useProcessImageMutation();
  const { data: history } = useGetImageHistoryQuery();

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFeatureToggle = (feature: string) => {
    setSelectedFeatures(prev => 
      prev.includes(feature)
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) {
      setError('请选择图片');
      return;
    }

    setError('');

    try {
      const result = await processImage({
        file: selectedFile,
        features: selectedFeatures
      }).unwrap();
      
      // Handle successful processing
      console.log('Processing result:', result);
      
      // You can add state management for the results here
      // For example, storing the processed images in state
      
    } catch (err: any) {
      setError(err.data?.detail || '处理失败，请重试');
    }
  };

  if (!isAuthenticated) {
    return <div>请先登录</div>;
  }

  return (
    <div className="text-analysis-container">
      <h2>图像处理</h2>
      <form onSubmit={handleSubmit} className="analysis-form">
        {error && <div className="error">{error}</div>}
        
        <div className="form-group">
          <label htmlFor="image">选择图片：</label>
          <input
            type="file"
            id="image"
            accept="image/*"
            onChange={handleFileSelect}
            disabled={isLoading}
            className="file-input"
          />
        </div>

        {preview && (
          <div className="preview-container">
            <h3>预览：</h3>
            <img src={preview} alt="Preview" className="image-preview" />
          </div>
        )}

        <div className="feature-selection">
          <h3>选择处理特征：</h3>
          <div className="feature-buttons">
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('enhancement') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('enhancement')}
              disabled={isLoading || !selectedFile}
            >
              图像增强
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('detection') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('detection')}
              disabled={isLoading || !selectedFile}
            >
              目标检测
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('segmentation') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('segmentation')}
              disabled={isLoading || !selectedFile}
            >
              图像分割
            </button>
            <button
              type="button"
              className={`feature-button ${selectedFeatures.includes('style') ? 'active' : ''}`}
              onClick={() => handleFeatureToggle('style')}
              disabled={isLoading || !selectedFile}
            >
              风格迁移
            </button>
          </div>
        </div>

        <button 
          type="submit" 
          disabled={isLoading || !selectedFile}
          className="submit-button"
        >
          {isLoading ? '处理中...' : '开始处理'}
        </button>
      </form>

      {isLoading && (
        <div className="loading-indicator">
          正在处理图片，请稍候...
        </div>
      )}

      {history && history.length > 0 && (
        <div className="history-section">
          <h3>处理历史</h3>
          <div className="history-grid">
            {history.map((item) => (
              <div key={item.id} className="history-item">
                <img src={item.image_url} alt="Original" className="history-image" />
                {item.result.enhanced && (
                  <div className="result-item">
                    <h4>增强结果</h4>
                    <img src={item.result.enhanced} alt="Enhanced" className="result-image" />
                  </div>
                )}
                {item.result.detections && (
                  <div className="result-item">
                    <h4>检测结果</h4>
                    <ul>
                      {item.result.detections.map((detection, index) => (
                        <li key={index}>
                          {detection.class} ({Math.round(detection.confidence * 100)}%)
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {item.result.segmentation && (
                  <div className="result-item">
                    <h4>分割结果</h4>
                    <p>分割数量: {item.result.segmentation.segments}</p>
                    <img src={item.result.segmentation.segmented} alt="Segmented" className="result-image" />
                  </div>
                )}
                {item.result.styled && (
                  <div className="result-item">
                    <h4>风格迁移结果</h4>
                    <img src={item.result.styled} alt="Styled" className="result-image" />
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

export default ImageProcessing; 