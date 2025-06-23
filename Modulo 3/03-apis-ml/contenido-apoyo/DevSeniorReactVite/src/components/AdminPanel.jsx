import React, { useState } from 'react';
import './AdminPanel.css';
import Slidebar from './Slidebar';

const AdminPanel = ({ isDarkTheme, onToggleTheme }) => {
  const [selectedModel, setSelectedModel] = useState(null);

  // Mock data for AI models
  const aiModels = [
    { id: 1, name: 'Modelo de Clasificación', status: 'Activo', accuracy: '95%' },
    { id: 2, name: 'Modelo de Predicción', status: 'Inactivo', accuracy: '88%' },
    { id: 3, name: 'Modelo de Detección', status: 'Activo', accuracy: '92%' },
  ];

  return (
    <div className="admin-panel">
      <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
      <div className="main-content" style={{ marginLeft: 260 }}>
        <header>
          <h1>Gestión de Modelos de IA</h1>
          <button className="add-model-btn">+ Nuevo Modelo</button>
        </header>

        <div className="models-grid">
          {aiModels.map((model) => (
            <div
              key={model.id}
              className="model-card"
              onClick={() => setSelectedModel(model)}
            >
              <h3>{model.name}</h3>
              <div className="model-status">
                <span className={`status ${model.status.toLowerCase()}`}>
                  {model.status}
                </span>
              </div>
              <div className="model-metrics">
                <p>Precisión: {model.accuracy}</p>
              </div>
            </div>
          ))}
        </div>

        {selectedModel && (
          <div className="model-details">
            <h2>Detalles del Modelo</h2>
            <div className="details-content">
              <p><strong>Nombre:</strong> {selectedModel.name}</p>
              <p><strong>Estado:</strong> {selectedModel.status}</p>
              <p><strong>Precisión:</strong> {selectedModel.accuracy}</p>
              <div className="action-buttons">
                <button className="edit-btn">Editar</button>
                <button className="delete-btn">Eliminar</button>
              </div>
            </div>
          </div>
        )}
      </div>

      <button
        className="theme-toggle-btn"
        onClick={onToggleTheme}
        title={isDarkTheme ? "Cambiar a tema claro" : "Cambiar a tema oscuro"}
      >
        {isDarkTheme ? '☀️' : '🌙'}
      </button>
    </div>
  );
};

export default AdminPanel; 