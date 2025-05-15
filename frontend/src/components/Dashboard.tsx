import React from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

const Dashboard: React.FC = () => {
  const username = useSelector((state: RootState) => state.auth.username);

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <h2>欢迎回来, {username}</h2>
        <p className="dashboard-subtitle">您的文本分析仪表板</p>
      </div>

      <div className="dashboard-stats">
        <div className="stat-card">
          <h3>今日分析</h3>
          <p className="stat-number">0</p>
        </div>
        <div className="stat-card">
          <h3>本周分析</h3>
          <p className="stat-number">0</p>
        </div>
        <div className="stat-card">
          <h3>总分析次数</h3>
          <p className="stat-number">0</p>
        </div>
      </div>

      <div className="dashboard-sections">
        <div className="dashboard-section">
          <h3>快速分析</h3>
          <div className="quick-analysis">
            <textarea
              placeholder="输入要分析的文本..."
              className="quick-analysis-input"
            />
            <button className="analyze-button">分析</button>
          </div>
        </div>

        <div className="dashboard-section">
          <h3>最近分析</h3>
          <div className="recent-analyses">
            <p className="no-data">暂无分析记录</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 