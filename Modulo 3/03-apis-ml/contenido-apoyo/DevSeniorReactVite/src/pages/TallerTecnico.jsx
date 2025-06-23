import React from 'react';
import Slidebar from '../components/Slidebar';
import '../layouts/PageLayout.css';

const TallerTecnico = ({ isDarkTheme, onToggleTheme }) => (
  <div className={`page ${isDarkTheme ? 'dark' : ''}`}>
    <Slidebar isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <main className="main-content">
      <div className="content-wrapper">
        <div className="header">
          <div>
            <h1>🧪 Taller Técnico: Modelos de Machine Learning</h1>
            <p className="subtitle">
              Aplicación práctica de regresión lineal y logística con APIs
            </p>
          </div>
          <button 
            className="theme-toggle" 
            onClick={onToggleTheme}
            title={isDarkTheme ? "Cambiar a tema claro" : "Cambiar a tema oscuro"}
          >
            {isDarkTheme ? "☀️" : "🌙"}
          </button>
        </div>
        <div className="content">
          <section className="section">
            <h2>🎯 Objetivos del Taller</h2>
            <div className="objectives-grid">
              <div className="objective-card">
                <h3>📊 Análisis de Datos</h3>
                <p>Explorar y preparar datasets reales para machine learning.</p>
              </div>
              <div className="objective-card">
                <h3>🤖 Modelos de ML</h3>
                <p>Implementar y comparar regresión lineal y logística.</p>
              </div>
              <div className="objective-card">
                <h3>💾 Serialización</h3>
                <p>Guardar modelos entrenados para su reutilización.</p>
              </div>
              <div className="objective-card">
                <h3>🌐 API REST</h3>
                <p>Conceptos para exponer modelos como servicios web.</p>
              </div>
            </div>
          </section>
          <section className="section">
            <h2>📁 Datasets del Taller</h2>
            <div className="objectives-grid">
              <div className="dataset-card">
                <h3>
                  <a href="https://www.kaggle.com/datasets/aariyan101/usa-housingcsv" target="_blank" rel="noopener noreferrer">
                    🏠 USA Housing Dataset
                  </a>
                </h3>
                <p><strong>Propósito:</strong> Predicción de precios de viviendas.</p>
                <p><strong>Características:</strong> Ingreso promedio, edad de la vivienda, número de habitaciones, dormitorios y población del área.</p>
                <p><strong>Modelo sugerido:</strong> Regresión Lineal.</p>
              </div>
              <div className="dataset-card">
                <h3>
                  <a href="https://www.kaggle.com/datasets/rakeshrau/social-network-ads" target="_blank" rel="noopener noreferrer">
                    📱 Social Network Ads Dataset
                  </a>
                </h3>
                <p><strong>Propósito:</strong> Predicción de compras en redes sociales.</p>
                <p><strong>Características:</strong> Edad y salario estimado del usuario.</p>
                <p><strong>Modelo sugerido:</strong> Regresión Logística.</p>
              </div>
            </div>
          </section>
          <section className="section">
            <h2>📝 Actividad del Taller</h2>
            <ul>
              <li>Explora ambos datasets y comprende sus variables.</li>
              <li>Plantea un caso de negocio para cada dataset.</li>
              <li>Define el objetivo de predicción para cada caso.</li>
              <li>Reflexiona sobre qué tipo de modelo es más adecuado para cada problema y por qué.</li>
              <li>Piensa cómo podrías exponer un modelo entrenado como un servicio web (API).</li>
            </ul>
          </section>
          <section className="section">
            <h2>🚦 Entregables</h2>
            <ul>
              <li>Un breve informe conceptual sobre el análisis de los datasets y la justificación de los modelos elegidos.</li>
              <li>Un diagrama o esquema de cómo sería el flujo de datos desde el dataset hasta la predicción vía API (no es necesario código).</li>
            </ul>
          </section>
          <section className="section">
            <h2>💡 Recomendaciones</h2>
            <ul>
              <li>Enfócate en el razonamiento y la justificación, no en la implementación técnica.</li>
              <li>Piensa en los retos de llevar un modelo de ML a producción.</li>
              <li>Considera la importancia de la documentación y la comunicación de resultados.</li>
              <li>Una vez que adquieras conocimientos de frontend, ¡vuelve a este taller! Un gran siguiente paso es construir una interfaz de usuario que consuma la API que has diseñado.</li>
            </ul>
          </section>
        </div>
      </div>
    </main>
  </div>
);

export default TallerTecnico; 