import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import AdminPanel from './components/AdminPanel'
import ArquitecturaRedes from './pages/ArquitecturaRedes'
import FuncionesActivacion from './pages/FuncionesActivacion'
import TensorFlowKeras from './pages/TensorFlowKeras'
import TallerPractico from './pages/TallerPractico'
import './App.css'

function App() {
  const [isDarkTheme, setIsDarkTheme] = useState(false);
  const [currentPage, setCurrentPage] = useState('');

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1);
      setCurrentPage(hash);
    };
    window.addEventListener('hashchange', handleHashChange);
    handleHashChange();
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    document.body.classList.toggle('dark-theme', isDarkTheme);
    document.body.classList.toggle('light-theme', !isDarkTheme);
  }, [isDarkTheme]);

  const onToggleTheme = () => {
    setIsDarkTheme(!isDarkTheme);
  };

  const renderPage = () => {
    switch (currentPage) {
      case '/arquitectura-redes':
        return <ArquitecturaRedes isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
      case '/funciones-activacion':
        return <FuncionesActivacion isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
      case '/tensorflow-keras':
        return <TensorFlowKeras isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
      case '/taller-practico':
        return <TallerPractico isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
      default:
        return <AdminPanel isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
    }
  };

  return (
    <div className="app">
      {renderPage()}
    </div>
  );
}

export default App
