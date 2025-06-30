import React from 'react';
import './Slidebar.css';

const links = [
  {
    href: '#/',
    label: 'Introducción a Redes Neuronales',
    icon: '🧠',
    match: ['', '/'],
  },
  {
    href: '#/arquitectura-redes',
    label: 'Arquitectura de Redes Neuronales',
    icon: '🏗️',
    match: ['/arquitectura-redes'],
  },
  {
    href: '#/funciones-activacion',
    label: 'Funciones de Activación',
    icon: '⚡',
    match: ['/funciones-activacion'],
  },
  {
    href: '#/tensorflow-keras',
    label: 'TensorFlow & Keras',
    icon: '🤖',
    match: ['/tensorflow-keras'],
  },
  {
    href: '#/taller-practico',
    label: 'Taller Práctico',
    icon: '🧪',
    match: ['/taller-practico'],
  },
];

function getActivePath() {
  const hash = window.location.hash.slice(1);
  return hash === '' ? '/' : hash;
}

const Slidebar = ({ open, onClose, isDarkTheme, alwaysVisible }) => {
  const activePath = getActivePath();
  const navStyle = {
    position: 'fixed',
    left: 0,
    top: 0,
    height: '100vh',
    zIndex: 2100,
    minWidth: 220,
    background: isDarkTheme ? 'var(--bg-secondary, #232136)' : 'var(--bg-secondary, #fff)',
    borderRight: isDarkTheme ? '1.5px solid #353552' : '1.5px solid #e5e7eb',
    boxShadow: '2px 0 12px rgba(0,0,0,0.07)',
    padding: '32px 0 0 0',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'stretch',
  };
  const ulStyle = {
    listStyle: 'none',
    padding: 0,
    margin: 0,
    width: '100%',
  };
  const liStyle = {
    margin: 0,
    padding: 0,
  };
  const linkStyle = isActive => ({
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '14px 32px',
    color: isActive ? '#a259f7' : (isDarkTheme ? '#fff' : '#232136'),
    background: isActive ? (isDarkTheme ? 'rgba(162,89,247,0.10)' : 'rgba(162,89,247,0.08)') : 'none',
    fontWeight: isActive ? 700 : 500,
    fontSize: 16,
    textDecoration: 'none',
    borderLeft: isActive ? '4px solid #a259f7' : '4px solid transparent',
    borderRadius: '0 20px 20px 0',
    transition: 'background 0.2s, color 0.2s',
    cursor: 'pointer',
  });

  const content = (
    <nav className={`slidebar open ${isDarkTheme ? 'dark' : ''}`} style={navStyle}>
      <div style={{ fontWeight: 800, fontSize: 22, letterSpacing: 1, color: '#a259f7', margin: '0 0 32px 32px', display: 'flex', alignItems: 'center', gap: 10 }}>
        <span>🧑‍💻</span> Panel Admin
      </div>
      <ul style={ulStyle}>
        {links.map(link => {
          const isActive = link.match.includes(activePath);
          return (
            <li key={link.href} style={liStyle}>
              <a href={link.href} style={linkStyle(isActive)}>
                <span style={{ fontSize: 20 }}>{link.icon}</span>
                {link.label}
              </a>
            </li>
          );
        })}
      </ul>
    </nav>
  );

  if (alwaysVisible) {
    return content;
  }
  return (
    <div className={`slidebar-overlay${open ? ' open' : ''}`} onClick={onClose}>
      <nav
        className={`slidebar${open ? ' open' : ''} ${isDarkTheme ? 'dark' : ''}`}
        onClick={e => e.stopPropagation()}
        style={navStyle}
      >
        <button className="close-btn" onClick={onClose} title="Cerrar menú">×</button>
        {content}
      </nav>
    </div>
  );
};

export default Slidebar; 