import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { logout } from '../store/slices/authSlice';
import '../styles/components.css';

const Navigation: React.FC = () => {
  const location = useLocation();
  const dispatch = useDispatch();
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);
  const username = useSelector((state: RootState) => state.auth.username);

  const handleLogout = () => {
    dispatch(logout());
  };

  return (
    <nav className="main-nav">
      <div className="nav-container">
        <div className="nav-brand">
          <Link to="/">AI分析系统</Link>
        </div>
        <div className="nav-links">
          {isAuthenticated ? (
            <>
              <Link 
                to="/nlp" 
                className={location.pathname === '/nlp' ? 'active' : ''}
              >
                NLP分析
              </Link>
              <Link 
                to="/image" 
                className={location.pathname === '/image' ? 'active' : ''}
              >
                图像处理
              </Link>
              <Link 
                to="/image-generation" 
                className={location.pathname === '/image-generation' ? 'active' : ''}
              >
                AI 绘画
              </Link>
              <div className="nav-user">
                <span className="username">欢迎, {username}</span>
                <button onClick={handleLogout} className="logout-button">
                  退出
                </button>
              </div>
            </>
          ) : (
            <>
              <Link 
                to="/login" 
                className={location.pathname === '/login' ? 'active' : ''}
              >
                登录
              </Link>
              <Link 
                to="/register" 
                className={location.pathname === '/register' ? 'active' : ''}
              >
                注册
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navigation; 