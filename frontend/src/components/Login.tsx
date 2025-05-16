import React, { useState } from 'react';
import { useLoginMutation } from '../store/api';
import { useDispatch } from 'react-redux';
import { setCredentials } from '../store/authSlice';
import { useNavigate, Link } from 'react-router-dom';

interface ValidationError {
  loc: string[];
  msg: string;
  type: string;
}

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string>('');
  const [login, { isLoading }] = useLoginMutation();
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!username.trim() || !password.trim()) {
      setError('请输入用户名和密码');
      return;
    }

    try {
      const response = await login({ username, password }).unwrap();
      dispatch(setCredentials({ token: response.access_token, username }));
      navigate('/nlp');
    } catch (err: any) {
      // Handle FastAPI validation errors
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
        setError('登录失败，请重试');
      }
    }
  };

  return (
    <div className="auth-container">
      <form onSubmit={handleSubmit} className="auth-form">
        <h2>登录</h2>
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
        <button type="submit" className="submit-button" disabled={isLoading}>
          {isLoading ? '登录中...' : '登录'}
        </button>
        <div className="auth-links">
          <Link to="/register">还没有账号？立即注册</Link>
        </div>
      </form>
    </div>
  );
};

export default Login; 