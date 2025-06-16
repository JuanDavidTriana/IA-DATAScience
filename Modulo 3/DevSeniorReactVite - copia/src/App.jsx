import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import AdminPanel from './components/AdminPanel'
import FundamentosML from './pages/FundamentosML'
import ModeloAServicio from './pages/ModeloAServicio'
import ApiRobusta from './pages/ApiRobusta'
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
      case '/fundamentos-ml':
        return <FundamentosML isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
      case '/modelo-a-servicio':
        return <ModeloAServicio isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
      case '/api-robusta':
        return <ApiRobusta isDarkTheme={isDarkTheme} onToggleTheme={onToggleTheme} />;
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
