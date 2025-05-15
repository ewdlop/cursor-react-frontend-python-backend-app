import React, { useState } from 'react';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { useGetUserProfileQuery, useGetUserStatsQuery, useChangePasswordMutation } from '../store/api';

const UserProfile: React.FC = () => {
  const username = useSelector((state: RootState) => state.auth.username);
  const [isEditing, setIsEditing] = useState(false);
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');

  const { data: profile } = useGetUserProfileQuery();
  const { data: stats } = useGetUserStatsQuery();
  const [changePassword] = useChangePasswordMutation();

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (newPassword !== confirmPassword) {
      setError('两次输入的密码不一致');
      return;
    }

    if (newPassword.length < 6) {
      setError('密码长度至少为6个字符');
      return;
    }

    try {
      await changePassword({ new_password: newPassword }).unwrap();
      setIsEditing(false);
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      setError(err.data?.detail || '修改密码失败');
    }
  };

  return (
    <div className="profile-container">
      <div className="profile-header">
        <h2>个人资料</h2>
      </div>

      <div className="profile-content">
        <div className="profile-section">
          <h3>基本信息</h3>
          <div className="profile-info">
            <div className="info-item">
              <label>用户名</label>
              <span>{username}</span>
            </div>
            <div className="info-item">
              <label>注册时间</label>
              <span>{profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : '未知'}</span>
            </div>
          </div>
        </div>

        <div className="profile-section">
          <h3>修改密码</h3>
          {!isEditing ? (
            <button 
              className="edit-button"
              onClick={() => setIsEditing(true)}
            >
              修改密码
            </button>
          ) : (
            <form onSubmit={handlePasswordChange} className="password-form">
              {error && <div className="error-message">{error}</div>}
              <div className="form-group">
                <label htmlFor="newPassword">新密码</label>
                <input
                  type="password"
                  id="newPassword"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="confirmPassword">确认密码</label>
                <input
                  type="password"
                  id="confirmPassword"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>
              <div className="form-actions">
                <button type="submit" className="save-button">
                  保存
                </button>
                <button 
                  type="button" 
                  className="cancel-button"
                  onClick={() => {
                    setIsEditing(false);
                    setNewPassword('');
                    setConfirmPassword('');
                    setError('');
                  }}
                >
                  取消
                </button>
              </div>
            </form>
          )}
        </div>

        <div className="profile-section">
          <h3>使用统计</h3>
          <div className="usage-stats">
            <div className="stat-item">
              <span className="stat-label">今日分析</span>
              <span className="stat-value">{stats?.today || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">本周分析</span>
              <span className="stat-value">{stats?.week || 0}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">总分析次数</span>
              <span className="stat-value">{stats?.total || 0}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserProfile; 