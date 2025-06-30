import React, { useState } from 'react';
import './AdminPanel.css';
import Slidebar from './Slidebar';

const AdminPanel = ({ isDarkTheme, onToggleTheme }) => {
  const [selectedConcept, setSelectedConcept] = useState(null);

  // Mock data for neural network concepts
  const neuralConcepts = [
    { id: 1, name: 'Neurona Biológica', status: 'Completado', difficulty: 'Básico' },
    { id: 2, name: 'Perceptrón', status: 'En Progreso', difficulty: 'Intermedio' },
    { id: 3, name: 'Backpropagation', status: 'Pendiente', difficulty: 'Avanzado' },
  ];

  return (
    <div className="admin-panel">
      <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
      <div className="main-content" style={{ marginLeft: 260 }}>
        <header>
          <h1>🧠 Introducción a las Redes Neuronales</h1>
          <p style={{ fontSize: '18px', color: '#666', marginTop: '10px' }}>
            De la Biología a la Inteligencia Artificial
          </p>
        </header>

        <div className="welcome-section" style={{ 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
          color: 'white', 
          padding: '40px', 
          borderRadius: '16px', 
          marginBottom: '40px',
          textAlign: 'center'
        }}>
          <h2 style={{ fontSize: '28px', marginBottom: '20px' }}>
            ¡Bienvenidos a esta introducción a las redes neuronales!
          </h2>
          <p style={{ fontSize: '18px', lineHeight: '1.6', maxWidth: '800px', margin: '0 auto' }}>
            Hoy vamos a desmitificar uno de los campos más emocionantes de la inteligencia artificial. 
            Veremos qué son, cómo funcionan y construiremos nuestra primera red neuronal.
          </p>
        </div>

        <div className="course-overview" style={{ 
          background: 'white', 
          padding: '30px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb',
          marginBottom: '40px'
        }}>
          <h3 style={{ color: '#667eea', marginBottom: '20px' }}>🎯 ¿Qué Aprenderás?</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            <div style={{ background: '#f0f9ff', padding: '20px', borderRadius: '8px', border: '1px solid #bae6fd' }}>
              <h4 style={{ color: '#0369a1', marginBottom: '10px' }}>🧠 Fundamentos Teóricos</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                <li>Arquitectura de redes neuronales densas</li>
                <li>Funciones de activación y su impacto</li>
                <li>Funciones de pérdida y optimización</li>
                <li>Proceso de entrenamiento y backpropagation</li>
              </ul>
            </div>
            <div style={{ background: '#f0fdf4', padding: '20px', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
              <h4 style={{ color: '#166534', marginBottom: '10px' }}>⚙️ Implementación Práctica</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                <li>Configuración de TensorFlow/Keras</li>
                <li>Construcción de modelos desde cero</li>
                <li>Preprocesamiento de datos</li>
                <li>Evaluación y validación de modelos</li>
              </ul>
            </div>
            <div style={{ background: '#fef3c7', padding: '20px', borderRadius: '8px', border: '1px solid #fde68a' }}>
              <h4 style={{ color: '#92400e', marginBottom: '10px' }}>🎯 Aplicaciones Reales</h4>
              <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                <li>Clasificación de imágenes</li>
                <li>Predicción de series temporales</li>
                <li>Análisis de sentimientos</li>
                <li>Reconocimiento de patrones</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="examples-section" style={{ 
          background: 'white', 
          padding: '30px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb',
          marginBottom: '40px'
        }}>
          <h3 style={{ color: '#667eea', marginBottom: '20px', fontSize: '22px' }}>🎯 Ejemplos de Uso en el Mundo Real</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px' }}>
            <div style={{ 
              padding: '20px', 
              background: 'linear-gradient(135deg, #ff6b6b, #ee5a24)', 
              color: 'white', 
              borderRadius: '10px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>📱</div>
              <h4 style={{ marginBottom: '10px' }}>Reconocimiento Facial</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>
                Desbloqueo de smartphones, seguridad biométrica, etiquetado automático en redes sociales
              </p>
            </div>
            <div style={{ 
              padding: '20px', 
              background: 'linear-gradient(135deg, #4ecdc4, #44a08d)', 
              color: 'white', 
              borderRadius: '10px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>🚗</div>
              <h4 style={{ marginBottom: '10px' }}>Vehículos Autónomos</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>
                Detección de objetos, navegación, toma de decisiones en tiempo real
              </p>
            </div>
            <div style={{ 
              padding: '20px', 
              background: 'linear-gradient(135deg, #45b7d1, #96c93d)', 
              color: 'white', 
              borderRadius: '10px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>🏥</div>
              <h4 style={{ marginBottom: '10px' }}>Diagnóstico Médico</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>
                Análisis de radiografías, detección de cáncer, predicción de enfermedades
              </p>
            </div>
            <div style={{ 
              padding: '20px', 
              background: 'linear-gradient(135deg, #f093fb, #f5576c)', 
              color: 'white', 
              borderRadius: '10px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>💬</div>
              <h4 style={{ marginBottom: '10px' }}>Procesamiento de Lenguaje</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>
                Traducción automática, chatbots, análisis de sentimientos
              </p>
            </div>
            <div style={{ 
              padding: '20px', 
              background: 'linear-gradient(135deg, #667eea, #764ba2)', 
              color: 'white', 
              borderRadius: '10px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>🎮</div>
              <h4 style={{ marginBottom: '10px' }}>Videojuegos</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>
                IA de enemigos, generación de contenido, optimización de gráficos
              </p>
            </div>
            <div style={{ 
              padding: '20px', 
              background: 'linear-gradient(135deg, #fa709a, #fee140)', 
              color: 'white', 
              borderRadius: '10px',
              textAlign: 'center'
            }}>
              <div style={{ fontSize: '40px', marginBottom: '10px' }}>💰</div>
              <h4 style={{ marginBottom: '10px' }}>Finanzas</h4>
              <p style={{ fontSize: '14px', lineHeight: '1.5' }}>
                Predicción de precios, detección de fraude, trading algorítmico
              </p>
            </div>
          </div>
        </div>

        <div className="concepts-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '40px' }}>
          <div className="concept-card" style={{ 
            background: 'white', 
            padding: '30px', 
            borderRadius: '12px', 
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e5e7eb'
          }}>
            <h3 style={{ color: '#667eea', marginBottom: '15px' }}>🔬 ¿Qué es una Red Neuronal?</h3>
            <p style={{ lineHeight: '1.6', color: '#374151' }}>
              Inspiradas en el cerebro humano, las redes neuronales artificiales son modelos computacionales 
              que aprenden a realizar tareas reconociendo patrones en los datos. No "piensan" como los humanos, 
              pero son increíblemente buenas para tareas específicas.
            </p>
          </div>

          <div className="concept-card" style={{ 
            background: 'white', 
            padding: '30px', 
            borderRadius: '12px', 
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e5e7eb'
          }}>
            <h3 style={{ color: '#667eea', marginBottom: '15px' }}>🧩 La Neurona Artificial</h3>
            <p style={{ lineHeight: '1.6', color: '#374151' }}>
              La unidad fundamental de una red neuronal es la neurona (o nodo). Al igual que una neurona 
              biológica recibe señales, las procesa y las transmite, una neurona artificial hace algo 
              similar con datos numéricos.
            </p>
          </div>

          <div className="concept-card" style={{ 
            background: 'white', 
            padding: '30px', 
            borderRadius: '12px', 
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e5e7eb'
          }}>
            <h3 style={{ color: '#667eea', marginBottom: '15px' }}>🏗️ Arquitectura Densa</h3>
            <p style={{ lineHeight: '1.6', color: '#374151' }}>
              La arquitectura más básica y común es la red neuronal densa (Dense Neural Network), 
              también conocida como Perceptrón Multicapa (MLP). Cada neurona está conectada a todas las neuronas de la siguiente capa.
            </p>
          </div>
        </div>

        <div className="comparison-section" style={{ 
          background: 'white', 
          padding: '30px', 
          borderRadius: '12px', 
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
          border: '1px solid #e5e7eb',
          marginBottom: '40px'
        }}>
          <h3 style={{ color: '#667eea', marginBottom: '20px', fontSize: '22px' }}>📊 Comparación con Otros Modelos de ML</h3>
          
          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px',
            overflowX: 'auto'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px', textAlign: 'center' }}>🔄 Tabla Comparativa: Redes Neuronales vs Modelos Lineales</h4>
            <table style={{ 
              width: '100%', 
              borderCollapse: 'collapse', 
              fontSize: '14px',
              minWidth: '800px'
            }}>
              <thead>
                <tr style={{ background: '#667eea', color: 'white' }}>
                  <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Característica</th>
                  <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Regresión Lineal</th>
                  <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Regresión Logística</th>
                  <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Redes Neuronales</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ background: '#f8f9fa' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>🎯 Tipo de Problema</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Regresión (valor continuo)</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Clasificación binaria</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Regresión y clasificación</td>
                </tr>
                <tr style={{ background: 'white' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>📈 Complejidad del Modelo</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muy simple</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Simple</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muy compleja</td>
                </tr>
                <tr style={{ background: '#f8f9fa' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>🧮 Ecuación Base</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center', fontFamily: 'monospace' }}>y = w₁x₁ + w₂x₂ + ... + b</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center', fontFamily: 'monospace' }}>p = 1/(1 + e^(-z))</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center', fontFamily: 'monospace' }}>Múltiples capas con activaciones</td>
                </tr>
                <tr style={{ background: 'white' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>⚡ Función de Activación</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Ninguna (lineal)</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Sigmoide</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>ReLU, Tanh, Softmax, etc.</td>
                </tr>
                <tr style={{ background: '#f8f9fa' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>📊 Capacidad de Aprendizaje</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Solo relaciones lineales</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Relaciones no lineales simples</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Relaciones muy complejas</td>
                </tr>
                <tr style={{ background: 'white' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>🎯 Interpretabilidad</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muy alta</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Alta</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Baja (caja negra)</td>
                </tr>
                <tr style={{ background: '#f8f9fa' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>⚙️ Hiperparámetros</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muy pocos</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Pocos</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muchos</td>
                </tr>
                <tr style={{ background: 'white' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>🚀 Velocidad de Entrenamiento</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muy rápida</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Rápida</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Lenta</td>
                </tr>
                <tr style={{ background: '#f8f9fa' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>💾 Requisitos de Datos</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Pocos datos</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Pocos datos</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Muchos datos</td>
                </tr>
                <tr style={{ background: 'white' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>🔧 Preprocesamiento</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Mínimo</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Escalado recomendado</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Normalización crítica</td>
                </tr>
                <tr style={{ background: '#f8f9fa' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>🎯 Casos de Uso Ideales</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Predicción de precios, tendencias</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Clasificación binaria simple</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Problemas complejos, imágenes, texto</td>
                </tr>
                <tr style={{ background: 'white' }}>
                  <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>⚠️ Limitaciones</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Solo relaciones lineales</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Solo clasificación binaria</td>
                  <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Overfitting, interpretabilidad</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div style={{ 
            background: '#e3f2fd', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #bbdefb',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#1976d2', marginBottom: '15px' }}>💡 ¿Cuándo Usar Cada Modelo?</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '15px' }}>
              <div style={{ background: '#e8f5e8', padding: '15px', borderRadius: '8px' }}>
                <h5 style={{ color: '#2d5a2d', marginBottom: '10px' }}>📈 Regresión Lineal</h5>
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                  <li>Predicción de precios de viviendas</li>
                  <li>Análisis de tendencias temporales</li>
                  <li>Relaciones simples entre variables</li>
                  <li>Cuando necesitas interpretabilidad</li>
                </ul>
              </div>
              <div style={{ background: '#fff3cd', padding: '15px', borderRadius: '8px' }}>
                <h5 style={{ color: '#856404', marginBottom: '10px' }}>🎯 Regresión Logística</h5>
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                  <li>Clasificación de emails (spam/no spam)</li>
                  <li>Predicción de riesgo crediticio</li>
                  <li>Diagnóstico médico binario</li>
                  <li>Cuando necesitas probabilidades</li>
                </ul>
              </div>
              <div style={{ background: '#fce4ec', padding: '15px', borderRadius: '8px' }}>
                <h5 style={{ color: '#ad1457', marginBottom: '10px' }}>🧠 Redes Neuronales</h5>
                <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                  <li>Reconocimiento de imágenes</li>
                  <li>Procesamiento de lenguaje natural</li>
                  <li>Predicciones complejas con muchos datos</li>
                  <li>Cuando la precisión es más importante que la interpretabilidad</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {selectedConcept && (
          <div className="concept-details" style={{ 
            background: 'white', 
            padding: '30px', 
            borderRadius: '12px', 
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            border: '1px solid #e5e7eb',
            marginTop: '20px'
          }}>
            <h2 style={{ color: '#667eea', marginBottom: '20px' }}>Detalles del Concepto</h2>
            <div className="details-content">
              <p><strong>Nombre:</strong> {selectedConcept.name}</p>
              <p><strong>Estado:</strong> {selectedConcept.status}</p>
              <p><strong>Dificultad:</strong> {selectedConcept.difficulty}</p>
              <div className="action-buttons" style={{ marginTop: '20px' }}>
                <button className="edit-btn" style={{ 
                  background: '#667eea', 
                  color: 'white', 
                  border: 'none', 
                  padding: '10px 20px', 
                  borderRadius: '6px', 
                  marginRight: '10px',
                  cursor: 'pointer'
                }}>Explorar</button>
                <button className="delete-btn" style={{ 
                  background: '#ef4444', 
                  color: 'white', 
                  border: 'none', 
                  padding: '10px 20px', 
                  borderRadius: '6px',
                  cursor: 'pointer'
                }}>Cerrar</button>
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