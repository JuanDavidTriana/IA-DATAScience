import React, { useState } from 'react';
import Slidebar from '../components/Slidebar';
import '../layouts/PageLayout.css';

const TallerPractico = ({ isDarkTheme, onToggleTheme }) => {
  const [activeExercise, setActiveExercise] = useState('heart-disease');

  const exercises = [
    {
      id: 'heart-disease',
      title: '🏥 Clasificación de Enfermedades Cardíacas',
      difficulty: 'Intermedio',
      duration: '45-60 min',
      description: 'Predicción de enfermedades cardíacas usando redes neuronales'
    },
    {
      id: 'student-performance',
      title: '📚 Predicción de Rendimiento Académico',
      difficulty: 'Intermedio',
      duration: '60-75 min',
      description: 'Predicción de calificaciones finales de estudiantes'
    }
  ];

  return (
    <div className={`page ${isDarkTheme ? 'dark' : ''}`}>
      <Slidebar isDarkTheme={isDarkTheme} alwaysVisible={true} />
      <main className="main-content">
        <div className="content-wrapper">
          <div className="header">
            <div>
              <h1>🧪 Taller Práctico: Redes Neuronales</h1>
              <p className="subtitle">
                Aplicación práctica de conceptos de redes neuronales con TensorFlow/Keras
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
            {/* Exercise Navigation */}
            <section className="section">
              <h2>📋 Ejercicios Disponibles</h2>
              <div className="objectives-grid">
                {exercises.map((exercise) => (
                  <div
                    key={exercise.id}
                    onClick={() => setActiveExercise(exercise.id)}
                    className="objective-card"
                    style={{ 
                      cursor: 'pointer',
                      border: activeExercise === exercise.id ? '2px solid #667eea' : '1px solid #e5e7eb',
                      background: activeExercise === exercise.id ? '#f0f9ff' : 'white'
                    }}
                  >
                    <h3>{exercise.title}</h3>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                      <span style={{ 
                        padding: '4px 8px', 
                        borderRadius: '4px', 
                        fontSize: '12px',
                        background: '#10b981',
                        color: 'white'
                      }}>
                        {exercise.difficulty}
                      </span>
                      <span style={{ fontSize: '12px', color: '#6b7280' }}>{exercise.duration}</span>
                    </div>
                    <p>{exercise.description}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Heart Disease Exercise */}
            {activeExercise === 'heart-disease' && (
              <>
                {/* Dataset Overview */}
                <section className="section">
                  <h2>🏥 Clasificación de Enfermedades Cardíacas</h2>
                  
                  <div className="objective-card" style={{ background: '#f8f9fa', border: '1px solid #e9ecef' }}>
                    <h3>📊 Dataset: Heart Disease Cleveland UCI</h3>
                    <p style={{ marginBottom: '15px', lineHeight: '1.6' }}>
                      Este dataset contiene información médica de pacientes para predecir la presencia de enfermedades cardíacas. 
                      Es un problema de clasificación binaria donde el objetivo es determinar si un paciente tiene enfermedad cardíaca (1) o no (0).
                    </p>
                    
                    <div className="objectives-grid">
                      <div className="objective-card" style={{ background: '#e3f2fd' }}>
                        <h4 style={{ color: '#1976d2' }}>📈 Características del Dataset</h4>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                          <li><strong>13 características</strong> médicas</li>
                          <li><strong>303 registros</strong> de pacientes</li>
                          <li><strong>Clasificación binaria</strong> (0/1)</li>
                          <li>Datos balanceados</li>
                        </ul>
                      </div>
                      <div className="objective-card" style={{ background: '#f3e5f5' }}>
                        <h4 style={{ color: '#7b1fa2' }}>🎯 Objetivo</h4>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                          <li>Predecir enfermedad cardíaca</li>
                          <li>Máxima precisión posible</li>
                          <li>Evitar falsos negativos</li>
                          <li>Interpretar resultados</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </section>

                {/* Features Description */}
                <section className="section">
                  <h3>🔍 Descripción de Características</h3>
                  <div className="objective-card" style={{ background: '#f8f9fa', border: '1px solid #e9ecef', overflowX: 'auto' }}>
                    <table style={{ 
                      width: '100%', 
                      borderCollapse: 'collapse', 
                      fontSize: '14px',
                      minWidth: '600px'
                    }}>
                      <thead>
                        <tr style={{ background: '#667eea', color: 'white' }}>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Característica</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Descripción</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Tipo</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Rango</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>age</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Edad en años</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>29-77</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>sex</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Sexo (1=male, 0=female)</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-1</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>cp</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tipo de dolor en el pecho</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-3</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>trestbps</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Presión arterial en reposo (mm Hg)</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>94-200</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>chol</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Colesterol sérico (mg/dl)</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>126-564</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>fbs</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Azúcar en sangre &gt; 120 mg/dl</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-1</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>restecg</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Resultados electrocardiográficos</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-2</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>thalach</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Frecuencia cardíaca máxima</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>71-202</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>exang</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Angina inducida por ejercicio</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-1</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>oldpeak</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Depresión ST inducida por ejercicio</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-6.2</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>slope</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Pendiente del segmento ST pico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-2</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>ca</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Número de vasos principales</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-4</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>thal</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tipo de defecto talámico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-3</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* Implementation Steps */}
                <section className="section">
                  <h3>⚙️ Pasos de Implementación</h3>
                  <div className="objectives-grid">
                    <div className="objective-card" style={{ background: '#e8f5e8', border: '1px solid #bbf7d0' }}>
                      <h4 style={{ color: '#166534' }}>1️⃣ Preparación de Datos</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Cargar y explorar el dataset</li>
                        <li>Verificar valores faltantes</li>
                        <li>Normalizar características numéricas</li>
                        <li>Codificar variables categóricas</li>
                        <li>Dividir en train/test (80/20)</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#fff3cd', border: '1px solid #fde68a' }}>
                      <h4 style={{ color: '#92400e' }}>2️⃣ Arquitectura de la Red</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Capa de entrada: 13 neuronas</li>
                        <li>Capa oculta 1: 64 neuronas (ReLU)</li>
                        <li>Capa oculta 2: 32 neuronas (ReLU)</li>
                        <li>Capa de salida: 1 neurona (Sigmoid)</li>
                        <li>Dropout para regularización</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#fce4ec', border: '1px solid #fbb6ce' }}>
                      <h4 style={{ color: '#ad1457' }}>3️⃣ Entrenamiento</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Función de pérdida: Binary Crossentropy</li>
                        <li>Optimizador: Adam (lr=0.001)</li>
                        <li>Batch size: 32</li>
                        <li>Épocas: 100 con early stopping</li>
                        <li>Validación: 20% del training</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#e3f2fd', border: '1px solid #bbdefb' }}>
                      <h4 style={{ color: '#1976d2' }}>4️⃣ Evaluación</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Matriz de confusión</li>
                        <li>Precisión, Recall, F1-Score</li>
                        <li>Curva ROC y AUC</li>
                        <li>Análisis de importancia de features</li>
                        <li>Comparación con modelos lineales</li>
                      </ul>
                    </div>
                  </div>
                </section>

                {/* Layer Selection Guide */}
                <section className="section">
                  <h3>🧠 Guía de Selección de Capas</h3>
                  <div className="objective-card" style={{ background: '#f8f9fa', border: '1px solid #e9ecef' }}>
                    <h4 style={{ color: '#495057', marginBottom: '15px' }}>¿Por qué esta arquitectura para clasificación binaria?</h4>
                    
                    <div className="objectives-grid">
                      <div className="objective-card" style={{ background: '#e3f2fd' }}>
                        <h5 style={{ color: '#1976d2' }}>📊 Capa de Entrada (13 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Una neurona por cada característica del dataset. 
                          En clasificación médica, cada variable (edad, presión arterial, colesterol, etc.) 
                          aporta información única para el diagnóstico.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#e8f5e8' }}>
                        <h5 style={{ color: '#166534' }}>🔗 Capa Oculta 1 (64 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Aproximadamente 5x el número de features de entrada. 
                          Permite capturar interacciones complejas entre variables médicas. 
                          El tamaño es suficiente para aprender patrones sin overfitting.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#fff3cd' }}>
                        <h5 style={{ color: '#92400e' }}>🔗 Capa Oculta 2 (32 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Reducción gradual (64 → 32) para comprimir 
                          información y extraer características más abstractas. 
                          Ayuda a generalizar mejor en datos médicos.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#fce4ec' }}>
                        <h5 style={{ color: '#ad1457' }}>🎯 Capa de Salida (1 neurona + Sigmoid)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Una neurona para clasificación binaria (enfermedad/no enfermedad). 
                          Sigmoid produce probabilidades entre 0-1, ideal para diagnóstico médico 
                          donde necesitas confianza en la predicción.
                        </p>
                      </div>
                    </div>
                  </div>
                </section>

                {/* Challenges and Extensions */}
                <section className="section">
                  <h3>🎯 Desafíos y Extensiones</h3>
                  <div className="objectives-grid">
                    <div className="objective-card" style={{ background: '#ffe6e6', border: '1px solid #fecaca' }}>
                      <h4 style={{ color: '#dc2626' }}>🔧 Optimización</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Experimentar con diferentes arquitecturas</li>
                        <li>Probar diferentes optimizadores</li>
                        <li>Ajustar hiperparámetros con GridSearch</li>
                        <li>Implementar cross-validation</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#e6ffe6', border: '1px solid #bbf7d0' }}>
                      <h4 style={{ color: '#16a34a' }}>📊 Análisis Avanzado</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Análisis de importancia de features</li>
                        <li>Visualización de activaciones</li>
                        <li>Interpretabilidad del modelo</li>
                        <li>Comparación con otros algoritmos</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#fff3cd', border: '1px solid #fde68a' }}>
                      <h4 style={{ color: '#92400e' }}>🚀 Despliegue</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Guardar y cargar el modelo</li>
                        <li>Crear API REST con Flask</li>
                        <li>Interfaz web para predicciones</li>
                        <li>Monitoreo de rendimiento</li>
                      </ul>
                    </div>
                  </div>
                </section>

                {/* Expected Results */}
                <section className="section">
                  <h3>📈 Resultados Esperados</h3>
                  <div className="objective-card" style={{ 
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                    color: 'white'
                  }}>
                    <div className="objectives-grid">
                      <div className="objective-card" style={{ background: 'rgba(255,255,255,0.1)' }}>
                        <h4>🎯 Precisión</h4>
                        <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>85-90%</p>
                      </div>
                      <div className="objective-card" style={{ background: 'rgba(255,255,255,0.1)' }}>
                        <h4>📊 AUC-ROC</h4>
                        <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>0.88-0.92</p>
                      </div>
                      <div className="objective-card" style={{ background: 'rgba(255,255,255,0.1)' }}>
                        <h4>⚡ Tiempo Entrenamiento</h4>
                        <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>2-5 min</p>
                      </div>
                    </div>
                  </div>
                </section>
              </>
            )}

            {/* Student Performance Exercise */}
            {activeExercise === 'student-performance' && (
              <>
                {/* Dataset Overview */}
                <section className="section">
                  <h2>📚 Predicción de Rendimiento Académico</h2>
                  
                  <div className="objective-card" style={{ background: '#f8f9fa', border: '1px solid #e9ecef' }}>
                    <h3>📊 Dataset: Student Alcohol Consumption</h3>
                    <p style={{ marginBottom: '15px', lineHeight: '1.6' }}>
                      Este dataset contiene información social, demográfica y académica de estudiantes de secundaria. 
                      El objetivo es predecir las calificaciones finales (G3) basándose en múltiples factores incluyendo 
                      consumo de alcohol, hábitos de estudio, apoyo familiar y características socioeconómicas.
                    </p>
                    
                    <div className="objectives-grid">
                      <div className="objective-card" style={{ background: '#e3f2fd' }}>
                        <h4 style={{ color: '#1976d2' }}>📈 Características del Dataset</h4>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                          <li><strong>33 características</strong> demográficas y académicas</li>
                          <li><strong>649 registros</strong> de estudiantes</li>
                          <li><strong>Problema de regresión</strong> (G3: 0-20)</li>
                          <li>Datos de Matemáticas y Portugués</li>
                        </ul>
                      </div>
                      <div className="objective-card" style={{ background: '#f3e5f5' }}>
                        <h4 style={{ color: '#7b1fa2' }}>🎯 Objetivo</h4>
                        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '14px' }}>
                          <li>Predecir calificación final (G3)</li>
                          <li>Identificar factores influyentes</li>
                          <li>Análisis de correlaciones</li>
                          <li>Modelo de regresión robusto</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </section>

                {/* Features Description */}
                <section className="section">
                  <h3>🔍 Descripción de Características</h3>
                  <div className="objective-card" style={{ background: '#f8f9fa', border: '1px solid #e9ecef', overflowX: 'auto' }}>
                    <table style={{ 
                      width: '100%', 
                      borderCollapse: 'collapse', 
                      fontSize: '14px',
                      minWidth: '600px'
                    }}>
                      <thead>
                        <tr style={{ background: '#667eea', color: 'white' }}>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Característica</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'left' }}>Descripción</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Tipo</th>
                          <th style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Rango/Valores</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>school</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Escuela del estudiante</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>GP, MS</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>sex</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Sexo del estudiante</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>F, M</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>age</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Edad del estudiante</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>15-22</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>address</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tipo de dirección</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>U, R</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>famsize</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tamaño de la familia</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>LE3, GT3</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Pstatus</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Estado de convivencia de padres</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>T, A</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Medu</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Educación de la madre</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-4</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Fedu</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Educación del padre</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-4</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Mjob</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Trabajo de la madre</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>teacher, health, services, at_home, other</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Fjob</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Trabajo del padre</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>teacher, health, services, at_home, other</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>reason</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Razón para elegir la escuela</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>home, reputation, course, other</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>guardian</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tutor del estudiante</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Categórico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>mother, father, other</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>traveltime</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tiempo de viaje a la escuela</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-4</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>studytime</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tiempo de estudio semanal</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-4</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>failures</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Número de fallas previas</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-4</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>schoolsup</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Apoyo educativo extra</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>famsup</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Apoyo familiar educativo</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>paid</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Clases pagadas extra</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>activities</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Actividades extra-curriculares</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>nursery</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Asistió a guardería</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>higher</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Quiere educación superior</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>internet</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Acceso a internet en casa</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>romantic</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Relación romántica</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Binario</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>yes, no</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>famrel</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Calidad de relaciones familiares</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-5</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>freetime</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Tiempo libre después de clases</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-5</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>goout</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Salir con amigos</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-5</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Dalc</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Consumo de alcohol en días laborales</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-5</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>Walc</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Consumo de alcohol en fines de semana</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-5</td>
                        </tr>
                        <tr style={{ background: '#f8f9fa' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>health</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Estado de salud actual</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>1-5</td>
                        </tr>
                        <tr style={{ background: 'white' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>absences</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Número de ausencias escolares</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-93</td>
                        </tr>
                        <tr style={{ background: '#e8f5e8' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>G1</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Calificación del primer período</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-20</td>
                        </tr>
                        <tr style={{ background: '#e8f5e8' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>G2</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Calificación del segundo período</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-20</td>
                        </tr>
                        <tr style={{ background: '#fce4ec' }}>
                          <td style={{ padding: '12px', border: '1px solid #ddd', fontWeight: 'bold' }}>G3</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd' }}>Calificación final (TARGET)</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>Numérico</td>
                          <td style={{ padding: '12px', border: '1px solid #ddd', textAlign: 'center' }}>0-20</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* Implementation Steps */}
                <section className="section">
                  <h3>⚙️ Pasos de Implementación</h3>
                  <div className="objectives-grid">
                    <div className="objective-card" style={{ background: '#e8f5e8', border: '1px solid #bbf7d0' }}>
                      <h4 style={{ color: '#166534' }}>1️⃣ Preparación de Datos</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Cargar y explorar ambos datasets (Math y Portugués)</li>
                        <li>Combinar datasets o trabajar por separado</li>
                        <li>Codificar variables categóricas</li>
                        <li>Normalizar características numéricas</li>
                        <li>Dividir en train/test (80/20)</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#fff3cd', border: '1px solid #fde68a' }}>
                      <h4 style={{ color: '#92400e' }}>2️⃣ Arquitectura de la Red</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Capa de entrada: 32 neuronas</li>
                        <li>Capa oculta 1: 128 neuronas (ReLU)</li>
                        <li>Capa oculta 2: 64 neuronas (ReLU)</li>
                        <li>Capa oculta 3: 32 neuronas (ReLU)</li>
                        <li>Capa de salida: 1 neurona (lineal)</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#fce4ec', border: '1px solid #fbb6ce' }}>
                      <h4 style={{ color: '#ad1457' }}>3️⃣ Entrenamiento</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Función de pérdida: Mean Squared Error</li>
                        <li>Optimizador: Adam (lr=0.001)</li>
                        <li>Batch size: 32</li>
                        <li>Épocas: 200 con early stopping</li>
                        <li>Validación: 20% del training</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#e3f2fd', border: '1px solid #bbdefb' }}>
                      <h4 style={{ color: '#1976d2' }}>4️⃣ Evaluación</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>MSE, MAE, RMSE</li>
                        <li>R² Score</li>
                        <li>Análisis de residuos</li>
                        <li>Importancia de features</li>
                        <li>Visualizaciones de predicciones</li>
                      </ul>
                    </div>
                  </div>
                </section>

                {/* Layer Selection Guide for Regression */}
                <section className="section">
                  <h3>🧠 Guía de Selección de Capas para Regresión</h3>
                  <div className="objective-card" style={{ background: '#f8f9fa', border: '1px solid #e9ecef' }}>
                    <h4 style={{ color: '#495057', marginBottom: '15px' }}>¿Por qué esta arquitectura para predicción de calificaciones?</h4>
                    
                    <div className="objectives-grid">
                      <div className="objective-card" style={{ background: '#e3f2fd' }}>
                        <h5 style={{ color: '#1976d2' }}>📊 Capa de Entrada (32 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Una neurona por cada característica del dataset. 
                          En predicción académica, cada variable (edad, tiempo de estudio, consumo de alcohol, etc.) 
                          puede influir en el rendimiento final.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#e8f5e8' }}>
                        <h5 style={{ color: '#166534' }}>🔗 Capa Oculta 1 (128 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Aproximadamente 4x el número de features. 
                          En regresión necesitamos más capacidad para capturar relaciones no lineales complejas 
                          entre variables sociales y académicas.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#fff3cd' }}>
                        <h5 style={{ color: '#92400e' }}>🔗 Capa Oculta 2 (64 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Reducción gradual (128 → 64) para comprimir 
                          información y extraer características más abstractas. 
                          Ayuda a generalizar mejor en datos educativos.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#fce4ec' }}>
                        <h5 style={{ color: '#ad1457' }}>🔗 Capa Oculta 3 (32 neuronas)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Capa adicional para capturar interacciones 
                          más sutiles. En regresión, las capas adicionales ayudan a modelar 
                          relaciones complejas entre variables.
                        </p>
                      </div>
                      
                      <div className="objective-card" style={{ background: '#f3e5f5' }}>
                        <h5 style={{ color: '#7b1fa2' }}>🎯 Capa de Salida (1 neurona + Lineal)</h5>
                        <p style={{ fontSize: '14px', lineHeight: '1.6' }}>
                          <strong>Justificación:</strong> Una neurona para regresión (calificación 0-20). 
                          Activación lineal permite valores continuos sin restricciones, 
                          ideal para predecir calificaciones exactas.
                        </p>
                      </div>
                    </div>
                  </div>
                </section>

                {/* Challenges and Extensions */}
                <section className="section">
                  <h3>🎯 Desafíos y Extensiones</h3>
                  <div className="objectives-grid">
                    <div className="objective-card" style={{ background: '#ffe6e6', border: '1px solid #fecaca' }}>
                      <h4 style={{ color: '#dc2626' }}>🔧 Optimización Avanzada</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Incluir G1 y G2 como features</li>
                        <li>Experimentar con diferentes arquitecturas</li>
                        <li>Usar técnicas de ensemble</li>
                        <li>Implementar cross-validation</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#e6ffe6', border: '1px solid #bbf7d0' }}>
                      <h4 style={{ color: '#16a34a' }}>📊 Análisis Comparativo</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Comparar rendimiento entre materias</li>
                        <li>Análisis de correlación con alcohol</li>
                        <li>Factores socioeconómicos</li>
                        <li>Impacto del apoyo familiar</li>
                      </ul>
                    </div>
                    <div className="objective-card" style={{ background: '#fff3cd', border: '1px solid #fde68a' }}>
                      <h4 style={{ color: '#92400e' }}>🚀 Aplicaciones Prácticas</h4>
                      <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
                        <li>Sistema de alerta temprana</li>
                        <li>Recomendaciones personalizadas</li>
                        <li>Dashboard de seguimiento</li>
                        <li>API para instituciones educativas</li>
                      </ul>
                    </div>
                  </div>
                </section>

                {/* Expected Results */}
                <section className="section">
                  <h3>📈 Resultados Esperados</h3>
                  <div className="objective-card" style={{ 
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                    color: 'white'
                  }}>
                    <div className="objectives-grid">
                      <div className="objective-card" style={{ background: 'rgba(255,255,255,0.1)' }}>
                        <h4>🎯 R² Score</h4>
                        <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>0.75-0.85</p>
                      </div>
                      <div className="objective-card" style={{ background: 'rgba(255,255,255,0.1)' }}>
                        <h4>📊 RMSE</h4>
                        <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>2.0-3.0</p>
                      </div>
                      <div className="objective-card" style={{ background: 'rgba(255,255,255,0.1)' }}>
                        <h4>⚡ Tiempo Entrenamiento</h4>
                        <p style={{ margin: 0, fontSize: '1.2rem', fontWeight: 'bold' }}>3-8 min</p>
                      </div>
                    </div>
                  </div>
                </section>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default TallerPractico; 