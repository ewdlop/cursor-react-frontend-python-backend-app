import React, { useState } from 'react';
import { useRegisterMutation } from '../store/api';
import { useNavigate, Link } from 'react-router-dom';

interface ValidationError {
  loc: string[];
  msg: string;
  type: string;
}

const Register: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState<string>('');
  const [register, { isLoading }] = useRegisterMutation();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!username.trim() || !password.trim() || !confirmPassword.trim()) {
      setError('请填写所有字段');
      return;
    }

    if (password !== confirmPassword) {
      setError('两次输入的密码不一致');
      return;
    }

    try {
      await register({ username, password }).unwrap();
      navigate('/login');
    } catch (err: any) {
      if (err.data?.detail && Array.isArray(err.data.detail)) {
        const validationErrors = err.data.detail as ValidationError[];
        setError(validationErrors.map(error => error.msg).join(', '));
      } else if (typeof err.data === 'string') {
        setError(err.data);
      } else if (err.data?.detail) {
        setError(err.data.detail);
      } else if (err.error) {
        setError(err.error);
      } else {
        setError('注册失败，请重试');
      }
    }
  };

  return (
    <div className="auth-container">
      <form onSubmit={handleSubmit} className="auth-form">
        <h2>注册</h2>
        {error && <div className="error">{error}</div>}
        <div className="form-group">
          <label htmlFor="username">用户名：</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">密码：</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="confirmPassword">确认密码：</label>
          <input
            type="password"
            id="confirmPassword"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="submit-button" disabled={isLoading}>
          {isLoading ? '注册中...' : '注册'}
        </button>
        <div className="auth-links">
          <Link to="/login">已有账号？立即登录</Link>
        </div>
      </form>
    </div>
  );
};

export default Register; 