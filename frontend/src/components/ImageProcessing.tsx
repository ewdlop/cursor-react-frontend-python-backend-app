import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

const ImageProcessing: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(['basic']);
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

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

    setIsLoading(true);
    setError('');

    // TODO: Implement image processing API call
    try {
      // const formData = new FormData();
      // formData.append('image', selectedFile);
      // const response = await processImage(formData).unwrap();
      // Handle response
    } catch (err: any) {
      setError(err.data?.detail || '处理失败，请重试');
    } finally {
      setIsLoading(false);
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

      {/* Results section will be added when API is implemented */}
    </div>
  );
};

export default ImageProcessing; 