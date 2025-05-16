import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { RootState } from '../store';

const Navigation: React.FC = () => {
  const location = useLocation();
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

  if (!isAuthenticated) {
    return null;
  }

  return (
    <nav className="main-nav">
      <div className="nav-container">
        <div className="nav-brand">
          <Link to="/">AI分析系统</Link>
        </div>
        <div className="nav-links">
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
        </div>
      </div>
    </nav>
  );
};

export default Navigation; 