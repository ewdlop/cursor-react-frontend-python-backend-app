import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { RootState } from './store';
import Navigation from './components/Navigation';
import TextAnalysis from './components/TextAnalysis';
import ImageProcessing from './components/ImageProcessing';
import ImageGeneration from './components/ImageGeneration';
import Login from './components/Login';
import Register from './components/Register';
import UserProfile from './components/UserProfile';
import TextGeneration from './components/TextGeneration';
import './styles/components.css';

const App: React.FC = () => {
  const isAuthenticated = useSelector((state: RootState) => state.auth.isAuthenticated);

  return (
    <Router>
      <div className="app">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/login" element={!isAuthenticated ? <Login /> : <Navigate to="/nlp" />} />
            <Route path="/register" element={!isAuthenticated ? <Register /> : <Navigate to="/nlp" />} />
            <Route path="/nlp" element={isAuthenticated ? <TextAnalysis /> : <Navigate to="/login" />} />
            <Route path="/image" element={isAuthenticated ? <ImageProcessing /> : <Navigate to="/login" />} />
            <Route path="/image-generation" element={isAuthenticated ? <ImageGeneration /> : <Navigate to="/login" />} />
            <Route path="/profile" element={isAuthenticated ? <UserProfile /> : <Navigate to="/login" />} />
            <Route path="/text-generation" element={isAuthenticated ? <TextGeneration /> : <Navigate to="/login" />} />
            <Route path="/" element={<Navigate to={isAuthenticated ? "/nlp" : "/login"} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
