import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { useProcessImageMutation, useGetImageHistoryQuery } from '../store/api';

const ImageProcessing: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(['basic']);
  const [error, setError] = useState<string>('');
  const [basicSettings, setBasicSettings] = useState({
    brightness: 1.0,
    hue: 1.0,
    saturation: 1.0,
    rotation: 0,
    flip: undefined as 'horizontal' | 'vertical' | undefined
  });
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

  const handleBasicSettingChange = (setting: string, value: number | string | undefined) => {
    setBasicSettings(prev => ({
      ...prev,
      [setting]: value
    }));
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
        features: selectedFeatures,
        ...basicSettings
      }).unwrap();
      
      console.log('Processing result:', result);
      setError('');
      
    } catch (err: any) {
      console.error('Processing error:', err);
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

        <div className="basic-settings">
          <h3>基本调整：</h3>
          <div className="settings-grid">
            <div className="setting-item">
              <label>亮度：</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={basicSettings.brightness}
                onChange={(e) => handleBasicSettingChange('brightness', parseFloat(e.target.value))}
                disabled={isLoading || !selectedFile}
              />
              <span>{basicSettings.brightness.toFixed(1)}</span>
            </div>
            
            <div className="setting-item">
              <label>色相：</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={basicSettings.hue}
                onChange={(e) => handleBasicSettingChange('hue', parseFloat(e.target.value))}
                disabled={isLoading || !selectedFile}
              />
              <span>{basicSettings.hue.toFixed(1)}</span>
            </div>
            
            <div className="setting-item">
              <label>饱和度：</label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={basicSettings.saturation}
                onChange={(e) => handleBasicSettingChange('saturation', parseFloat(e.target.value))}
                disabled={isLoading || !selectedFile}
              />
              <span>{basicSettings.saturation.toFixed(1)}</span>
            </div>
            
            <div className="setting-item">
              <label>旋转：</label>
              <input
                type="range"
                min="0"
                max="360"
                step="1"
                value={basicSettings.rotation}
                onChange={(e) => handleBasicSettingChange('rotation', parseInt(e.target.value))}
                disabled={isLoading || !selectedFile}
              />
              <span>{basicSettings.rotation}°</span>
            </div>
            
            <div className="setting-item">
              <label>翻转：</label>
              <select
                value={basicSettings.flip || ''}
                onChange={(e) => handleBasicSettingChange('flip', e.target.value || undefined)}
                disabled={isLoading || !selectedFile}
              >
                <option value="">无</option>
                <option value="horizontal">水平</option>
                <option value="vertical">垂直</option>
              </select>
            </div>
          </div>
        </div>

        <div className="feature-selection">
          <h3>高级处理：</h3>
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
                
                {/* Basic Processing Results */}
                {item.result.brightness && (
                  <div className="result-item">
                    <h4>亮度调整</h4>
                    <img src={item.result.brightness} alt="Brightness" className="result-image" />
                  </div>
                )}
                {item.result.hue && (
                  <div className="result-item">
                    <h4>色相调整</h4>
                    <img src={item.result.hue} alt="Hue" className="result-image" />
                  </div>
                )}
                {item.result.saturation && (
                  <div className="result-item">
                    <h4>饱和度调整</h4>
                    <img src={item.result.saturation} alt="Saturation" className="result-image" />
                  </div>
                )}
                {item.result.rotation && (
                  <div className="result-item">
                    <h4>旋转</h4>
                    <img src={item.result.rotation} alt="Rotation" className="result-image" />
                  </div>
                )}
                {item.result.flip && (
                  <div className="result-item">
                    <h4>翻转</h4>
                    <img src={item.result.flip} alt="Flip" className="result-image" />
                  </div>
                )}

                {/* Advanced Processing Results */}
                {item.result.enhanced && (
                  <div className="result-item">
                    <h4>增强结果</h4>
                    <img src={item.result.enhanced} alt="Enhanced" className="result-image" />
                  </div>
                )}
                {item.result.detections && item.result.detections.length > 0 && (
                  <div className="result-item">
                    <h4>检测结果</h4>
                    <ul className="detection-list">
                      {item.result.detections.map((detection, index) => (
                        <li key={index} className="detection-item">
                          <span className="detection-class">{detection.class}</span>
                          <span className="detection-confidence">
                            {Math.round(detection.confidence * 100)}%
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {item.result.segmentation && (
                  <div className="result-item">
                    <h4>分割结果</h4>
                    <div className="segmentation-results">
                      <div className="segmentation-main">
                        <div className="segmentation-image">
                          <h5>分割图像</h5>
                          <img src={item.result.segmentation.segmented} alt="Segmented" className="result-image" />
                        </div>
                        <div className="segmentation-mask">
                          <h5>分割掩码</h5>
                          <img src={item.result.segmentation.mask} alt="Mask" className="result-image" />
                        </div>
                      </div>
                      
                      <div className="segmentation-stats">
                        <p>总分割数量: {item.result.segmentation.segments}</p>
                      </div>
                      
                      {item.result.segmentation.top_segments && (
                        <div className="top-segments">
                          <h5>前10个最大分割区域</h5>
                          <div className="segments-grid">
                            {item.result.segmentation.top_segments.map((segment) => (
                              <div key={segment.id} className="segment-item">
                                <img src={segment.image} alt={`Segment ${segment.id}`} className="segment-image" />
                                <div className="segment-info">
                                  <p>区域 {segment.id}</p>
                                  <p>面积: {Math.round(segment.area)} 像素</p>
                                  <p>周长: {Math.round(segment.perimeter)} 像素</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
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