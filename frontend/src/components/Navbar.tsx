import React, { useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState } from '../store';
import { logout } from '../store/authSlice';
import { useNavigate } from 'react-router-dom';

const Navbar: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);
  const username = useSelector((state: RootState) => state.auth.username);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <h1>NLP 分析系统</h1>
      </div>
      
      <button 
        className="menu-toggle"
        onClick={() => setIsMenuOpen(!isMenuOpen)}
      >
        <span className="menu-icon"></span>
      </button>

      <div className={`navbar-menu ${isMenuOpen ? 'is-active' : ''}`}>
        {isAuthenticated ? (
          <>
            <div className="navbar-item">
              <span className="welcome-text">欢迎, {username}</span>
            </div>
            <div className="navbar-item">
              <button className="nav-button" onClick={() => navigate('/')}>
                仪表板
              </button>
            </div>
            <div className="navbar-item">
              <button className="nav-button" onClick={() => navigate('/analysis')}>
                文本分析
              </button>
            </div>
            <div className="navbar-item">
              <button className="nav-button logout" onClick={handleLogout}>
                退出登录
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="navbar-item">
              <button className="nav-button" onClick={() => navigate('/login')}>
                登录
              </button>
            </div>
            <div className="navbar-item">
              <button className="nav-button" onClick={() => navigate('/register')}>
                注册
              </button>
            </div>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar; 