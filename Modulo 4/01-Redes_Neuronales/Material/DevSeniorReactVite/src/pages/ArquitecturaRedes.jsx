import React from 'react';
import Slidebar from '../components/Slidebar';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const sectionCard = {
  background: 'var(--bg-secondary, #fff)',
  borderRadius: 16,
  boxShadow: '0 2px 12px rgba(0,0,0,0.07)',
  padding: 32,
  marginBottom: 32,
  border: '1px solid #e5e7eb',
  maxWidth: 1100,
  width: '100%',
  marginLeft: 'auto',
  marginRight: 'auto',
};

const accentTitle = {
  color: '#a259f7',
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  fontWeight: 700,
  fontSize: 24,
  marginBottom: 10,
};

const subtitle = {
  color: '#16a34a',
  fontWeight: 600,
  fontSize: 18,
  margin: '18px 0 8px 0',
  display: 'flex',
  alignItems: 'center',
  gap: 8,
};

const ArquitecturaRedes = ({ isDarkTheme, onToggleTheme }) => (
  <div className="admin-panel">
    <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <div className="main-content" style={{ marginLeft: 260, maxWidth: '100%', background: 'transparent', minHeight: '100vh', paddingBottom: 60 }}>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="building">🏗️</span> Arquitectura de Redes Neuronales
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Introducción: Fundamentos de Redes Neuronales Densas</b><br/>
          La arquitectura más básica y común de una red neuronal es la red neuronal densa (Dense Neural Network), 
          también conocida como Perceptrón Multicapa (MLP). En una red densa, cada neurona de una capa está 
          conectada a todas las neuronas de la capa siguiente.<br/><br/>

          <div style={subtitle}>🏛️ Estructura de una Red Neuronal</div>
          Una red neuronal se organiza en capas, cada una con un propósito específico:<br/><br/>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>📥 Capa de Entrada (Input Layer)</h4>
            <p style={{ margin: 0, lineHeight: '1.6' }}>
              Recibe los datos iniciales. El número de neuronas en esta capa es igual al número de características 
              (features) de nuestro conjunto de datos. Por ejemplo, si queremos predecir el precio de una casa 
              basándonos en su tamaño y número de habitaciones, nuestra capa de entrada tendría 2 neuronas.
            </p>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>🔍 Capas Ocultas (Hidden Layers)</h4>
            <p style={{ margin: 0, lineHeight: '1.6' }}>
              Son las capas intermedias entre la entrada y la salida. Aquí es donde ocurre la mayor parte del "aprendizaje". 
              Una red puede tener cero o muchas capas ocultas. La cantidad de neuronas en estas capas es una decisión 
              de diseño (un hiperparámetro).
            </p>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>📤 Capa de Salida (Output Layer)</h4>
            <p style={{ margin: 0, lineHeight: '1.6' }}>
              Produce el resultado final. El número de neuronas depende de la tarea:
            </p>
            <ul style={{ margin: '10px 0 0 20px' }}>
              <li><b>Regresión:</b> Generalmente una sola neurona (predecir un valor continuo, como un precio)</li>
              <li><b>Clasificación Binaria:</b> Una sola neurona con activación especial (predecir entre dos clases)</li>
              <li><b>Clasificación Multiclase:</b> Tantas neuronas como clases haya</li>
            </ul>
          </div>

          <div style={subtitle}>🎨 Visualización de la Arquitectura</div>
          <div style={{ 
            textAlign: 'center', 
            background: '#f8f9fa', 
            padding: '30px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px',
            fontFamily: 'monospace',
            fontSize: '16px'
          }}>
            <div style={{ marginBottom: '20px' }}>
              <strong>(Capa de Entrada)</strong>      <strong>(Capa Oculta)</strong>      <strong>(Capa de Salida)</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '40px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ width: '20px', height: '20px', background: '#667eea', borderRadius: '50%' }}></div>
                <div style={{ width: '20px', height: '20px', background: '#667eea', borderRadius: '50%' }}></div>
                <div style={{ width: '20px', height: '20px', background: '#667eea', borderRadius: '50%' }}></div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ width: '20px', height: '20px', background: '#16a34a', borderRadius: '50%' }}></div>
                <div style={{ width: '20px', height: '20px', background: '#16a34a', borderRadius: '50%' }}></div>
                <div style={{ width: '20px', height: '20px', background: '#16a34a', borderRadius: '50%' }}></div>
                <div style={{ width: '20px', height: '20px', background: '#16a34a', borderRadius: '50%' }}></div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ width: '20px', height: '20px', background: '#dc2626', borderRadius: '50%' }}></div>
              </div>
            </div>
            <div style={{ marginTop: '20px', fontSize: '14px', color: '#666' }}>
              Cada conexión tiene un peso (w) asociado que se ajusta durante el entrenamiento
            </div>
          </div>

          <div style={subtitle}>⚖️ Pesos y Sesgos</div>
          Cada conexión entre neuronas tiene un peso (w) asociado, que se ajusta durante el entrenamiento. 
          Estos pesos determinan la importancia de una señal. Además, cada neurona tiene un sesgo (b), 
          que es un valor que permite ajustar la salida de la neurona independientemente de sus entradas.<br/><br/>

          <div style={subtitle}>🧮 Cálculo dentro de una Neurona</div>
          Dentro de una neurona, el cálculo es simple:<br/><br/>

          <ol style={{ marginLeft: '20px', marginBottom: '20px' }}>
            <li>Se suman todas las entradas, cada una multiplicada por su peso</li>
            <li>Se añade el sesgo</li>
            <li>El resultado pasa a través de una función de activación</li>
          </ol>

          <div style={{ 
            textAlign: 'center', 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px',
            fontSize: '18px',
            fontFamily: 'monospace'
          }}>
            z = (w₁ × x₁ + w₂ × x₂ + ...) + b<br/>
            y = f(z)
          </div>

          <div style={subtitle}>🎯 Ejemplo Práctico: Predicción de Precios</div>
          Imaginemos que queremos predecir el precio de una casa basándonos en tres características:<br/><br/>

          <div style={{ 
            background: '#e8f5e8', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #c3e6c3',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#2d5a2d', marginBottom: '15px' }}>🏠 Características de Entrada:</h4>
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              <li><b>x₁:</b> Tamaño de la casa (m²)</li>
              <li><b>x₂:</b> Número de habitaciones</li>
              <li><b>x₃:</b> Año de construcción</li>
            </ul>
          </div>

          <div style={{ 
            background: '#fff3cd', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #ffeaa7',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#856404', marginBottom: '15px' }}>🧮 Proceso de Cálculo:</h4>
            <p style={{ margin: 0, lineHeight: '1.6' }}>
              Para una casa de 150m², 3 habitaciones, construida en 2010:<br/>
              z = w₁(150) + w₂(3) + w₃(2010) + b<br/>
              y = f(z) = precio predicho
            </p>
          </div>

          <div style={subtitle}>🔧 Hiperparámetros de Arquitectura</div>
          Los hiperparámetros son configuraciones que definen la estructura de la red:<br/><br/>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#495057', marginBottom: '10px' }}>📏 Número de Capas</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Redes superficiales: 1-2 capas ocultas</li>
                <li>Redes profundas: 3+ capas ocultas</li>
                <li>Más capas = más capacidad de aprendizaje</li>
              </ul>
            </div>
            <div style={{ background: '#f8f9fa', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#495057', marginBottom: '10px' }}>🔢 Neuronas por Capa</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Demasiadas = overfitting</li>
                <li>Muy pocas = underfitting</li>
                <li>Regla general: entre entrada y salida</li>
              </ul>
            </div>
          </div>

          <div style={subtitle}>⚠️ Consideraciones de Diseño</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#ffe6e6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#dc2626', marginBottom: '10px' }}>❌ Problemas Comunes</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Overfitting (sobreajuste)</li>
                <li>Underfitting (subajuste)</li>
                <li>Vanishing/Exploding gradients</li>
                <li>Curse of dimensionality</li>
              </ul>
            </div>
            <div style={{ background: '#e6ffe6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#16a34a', marginBottom: '10px' }}>✅ Soluciones</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Regularización (Dropout, L1/L2)</li>
                <li>Normalización de datos</li>
                <li>Inicialización adecuada de pesos</li>
                <li>Validación cruzada</li>
              </ul>
            </div>
          </div>

          <div style={subtitle}>🔬 7. Limitaciones y Desafíos</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#ffe6e6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#dc2626', marginBottom: '10px' }}>❌ Problemas Comunes</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Overfitting (sobreajuste)</li>
                <li>Underfitting (subajuste)</li>
                <li>Vanishing/Exploding gradients</li>
                <li>Curse of dimensionality</li>
              </ul>
            </div>
            <div style={{ background: '#e6ffe6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#16a34a', marginBottom: '10px' }}>✅ Soluciones</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Regularización (Dropout, L1/L2)</li>
                <li>Normalización de datos</li>
                <li>Inicialización adecuada de pesos</li>
                <li>Validación cruzada</li>
              </ul>
            </div>
          </div>

          <div style={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
            color: 'white', 
            padding: '20px', 
            borderRadius: '12px',
            marginTop: '30px'
          }}>
            <h3 style={{ marginBottom: '15px' }}>💡 Puntos Clave</h3>
            <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
              <li>La arquitectura densa es la base de las redes neuronales modernas</li>
              <li>Cada capa tiene un propósito específico en el procesamiento</li>
              <li>Los pesos y sesgos se aprenden durante el entrenamiento</li>
              <li>El diseño de la arquitectura es crucial para el rendimiento</li>
              <li>La complejidad debe balancearse con la capacidad de generalización</li>
            </ul>
          </div>
        </div>
      </div>
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

export default ArquitecturaRedes; 