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

const FuncionesActivacion = ({ isDarkTheme, onToggleTheme }) => (
  <div className="admin-panel">
    <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <div className="main-content" style={{ marginLeft: 260, maxWidth: '100%', background: 'transparent', minHeight: '100vh', paddingBottom: 60 }}>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="lightning">⚡</span> Funciones de Activación
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Introducción: ¿Por qué necesitamos funciones de activación?</b><br/>
          Si las neuronas solo sumaran y ponderaran, toda la red sería simplemente una gran ecuación lineal, 
          lo que limitaría su capacidad para aprender patrones complejos. Las funciones de activación introducen 
          no-linealidad, permitiendo a la red aprender relaciones mucho más sofisticadas.<br/><br/>

          <div style={subtitle}>🎯 Propósito de las Funciones de Activación</div>
          <ul style={{ marginLeft: '20px', marginBottom: '20px' }}>
            <li><b>Introducir No-linealidad:</b> Permiten que la red aprenda patrones complejos</li>
            <li><b>Controlar el Rango de Salida:</b> Limitan los valores de salida a rangos específicos</li>
            <li><b>Resolver el Problema del Gradiente:</b> Ayudan con el entrenamiento de redes profundas</li>
            <li><b>Interpretabilidad:</b> Algunas funciones producen salidas con significado específico</li>
          </ul>

          <div style={subtitle}>🔧 Funciones de Activación Comunes</div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>⚡ ReLU (Rectified Linear Unit)</h4>
            <p style={{ margin: '0 0 15px 0', lineHeight: '1.6' }}>
              Es la más utilizada en las capas ocultas. Su fórmula es muy simple: f(x) = max(0, x). 
              Si la entrada es positiva, la devuelve; si es negativa, devuelve cero.
            </p>
            <div style={{ 
              textAlign: 'center', 
              background: '#e9ecef', 
              padding: '15px', 
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '16px'
            }}>
              f(x) = max(0, x)
            </div>
            <div style={{ marginTop: '15px' }}>
              <strong>Ventajas:</strong> Computacionalmente eficiente, resuelve el problema del gradiente desvanecido<br/>
              <strong>Desventajas:</strong> Neurona "muerta" (siempre 0 para entradas negativas)
            </div>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>📈 Sigmoide</h4>
            <p style={{ margin: '0 0 15px 0', lineHeight: '1.6' }}>
              Comprime cualquier valor de entrada en un rango entre 0 y 1. Es útil en la capa de salida 
              para problemas de clasificación binaria, ya que la salida puede interpretarse como una probabilidad.
            </p>
            <div style={{ 
              textAlign: 'center', 
              background: '#e9ecef', 
              padding: '15px', 
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '16px'
            }}>
              f(x) = 1 / (1 + e^(-x))
            </div>
            <div style={{ marginTop: '15px' }}>
              <strong>Ventajas:</strong> Salida interpretable como probabilidad, derivada simple<br/>
              <strong>Desventajas:</strong> Gradiente desvanecido para valores extremos, no centrada en cero
            </div>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>🎲 Softmax</h4>
            <p style={{ margin: '0 0 15px 0', lineHeight: '1.6' }}>
              Se utiliza exclusivamente en la capa de salida para problemas de clasificación multiclase. 
              Toma las salidas de todas las neuronas de la capa y las convierte en una distribución de probabilidad, 
              donde la suma de todas las salidas es 1.
            </p>
            <div style={{ 
              textAlign: 'center', 
              background: '#e9ecef', 
              padding: '15px', 
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '16px'
            }}>
              f(x_i) = e^(x_i) / Σ(e^(x_j))
            </div>
            <div style={{ marginTop: '15px' }}>
              <strong>Ventajas:</strong> Distribución de probabilidad válida, interpretable<br/>
              <strong>Desventajas:</strong> Computacionalmente costosa, sensible a valores extremos
            </div>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>🟡 Tanh (Tangente Hiperbólica)</h4>
            <p style={{ margin: '0 0 15px 0', lineHeight: '1.6' }}>
              Similar a la sigmoide pero centrada en cero, con rango entre -1 y 1. 
              Útil cuando se necesitan valores negativos y positivos.
            </p>
            <div style={{ 
              textAlign: 'center', 
              background: '#e9ecef', 
              padding: '15px', 
              borderRadius: '6px',
              fontFamily: 'monospace',
              fontSize: '16px'
            }}>
              f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
            </div>
            <div style={{ marginTop: '15px' }}>
              <strong>Ventajas:</strong> Centrada en cero, gradientes más estables<br/>
              <strong>Desventajas:</strong> Gradiente desvanecido para valores extremos
            </div>
          </div>

          <div style={subtitle}>📊 Comparación Visual</div>
          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px',
            textAlign: 'center'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>Rangos de Salida</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
              <div style={{ background: '#e3f2fd', padding: '10px', borderRadius: '6px' }}>
                <strong>ReLU:</strong> [0, ∞)
              </div>
              <div style={{ background: '#f3e5f5', padding: '10px', borderRadius: '6px' }}>
                <strong>Sigmoide:</strong> (0, 1)
              </div>
              <div style={{ background: '#e8f5e8', padding: '10px', borderRadius: '6px' }}>
                <strong>Tanh:</strong> (-1, 1)
              </div>
              <div style={{ background: '#fff3e0', padding: '10px', borderRadius: '6px' }}>
                <strong>Softmax:</strong> (0, 1) con suma = 1
              </div>
            </div>
          </div>

          <div style={subtitle}>🎯 Cuándo Usar Cada Función</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#e8f5e8', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#2d5a2d', marginBottom: '10px' }}>✅ Capas Ocultas</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li><strong>ReLU:</strong> Opción por defecto</li>
                <li><strong>Leaky ReLU:</strong> Si hay neuronas muertas</li>
                <li><strong>Tanh:</strong> Si necesitas valores negativos</li>
                <li><strong>ELU:</strong> Alternativa moderna a ReLU</li>
              </ul>
            </div>
            <div style={{ background: '#fff3cd', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#856404', marginBottom: '10px' }}>🎯 Capa de Salida</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li><strong>Sigmoide:</strong> Clasificación binaria</li>
                <li><strong>Softmax:</strong> Clasificación multiclase</li>
                <li><strong>Lineal:</strong> Regresión</li>
                <li><strong>ReLU:</strong> Valores positivos</li>
              </ul>
            </div>
          </div>

          <div style={subtitle}>⚠️ Problemas Comunes</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#ffe6e6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#dc2626', marginBottom: '10px' }}>❌ Gradiente Desvanecido</h4>
              <p style={{ margin: 0, lineHeight: '1.6' }}>
                Ocurre cuando las derivadas de las funciones de activación se acercan a cero, 
                dificultando el entrenamiento de redes profundas. Afecta principalmente a Sigmoide y Tanh.
              </p>
            </div>
            <div style={{ background: '#ffe6e6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#dc2626', marginBottom: '10px' }}>💀 Neurona Muerta</h4>
              <p style={{ margin: 0, lineHeight: '1.6' }}>
                En ReLU, si una neurona siempre recibe entradas negativas, su gradiente será siempre cero 
                y nunca se actualizará. Solución: Leaky ReLU o ELU.
              </p>
            </div>
          </div>

          <div style={subtitle}>🔧 Funciones Modernas</div>
          
          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>🚀 Variantes de ReLU</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
              <div>
                <strong>Leaky ReLU:</strong> f(x) = max(αx, x) donde α es pequeño (ej: 0.01)
              </div>
              <div>
                <strong>ELU:</strong> f(x) = x si x > 0, α(e^x - 1) si x ≤ 0
              </div>
              <div>
                <strong>PReLU:</strong> Como Leaky ReLU pero α se aprende
              </div>
              <div>
                <strong>Swish:</strong> f(x) = x / (1 + e^(-x))
              </div>
            </div>
          </div>

          <div style={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
            color: 'white', 
            padding: '20px', 
            borderRadius: '12px',
            marginTop: '30px'
          }}>
            <h3 style={{ marginBottom: '15px' }}>💡 Recomendaciones Prácticas</h3>
            <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
              <li>Usa <strong>ReLU</strong> como función por defecto en capas ocultas</li>
              <li>Para clasificación binaria, usa <strong>Sigmoide</strong> en la salida</li>
              <li>Para clasificación multiclase, usa <strong>Softmax</strong> en la salida</li>
              <li>Si tienes problemas con neuronas muertas, prueba <strong>Leaky ReLU</strong></li>
              <li>Experimenta con diferentes funciones para tu problema específico</li>
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

export default FuncionesActivacion; 