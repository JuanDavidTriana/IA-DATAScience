import React from 'react';
import Slidebar from '../components/Slidebar';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const codePipeline = `# -*- coding: utf-8 -*-

# --- Importaciones Necesarias ---
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib # Biblioteca para guardar y cargar modelos

# --- 1. Cargar y Dividir los Datos ---
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 2. Definir los Pasos del Pipeline ---
pipeline_pasos = [
    ('escalador', StandardScaler()),
    ('clasificador', LogisticRegression(max_iter=200))
]

# --- 3. Crear y Entrenar el Pipeline ---
modelo_pipeline = Pipeline(pipeline_pasos)
print("Entrenando el pipeline completo...")
modelo_pipeline.fit(X_train, y_train)
print("¡Entrenamiento completado!\n")

# --- 4. Evaluar el Pipeline ---
y_pred = modelo_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"La exactitud del pipeline en el conjunto de prueba es: {accuracy:.2%}")

# --- 5. Guardar el Pipeline Entrenado para Producción ---
nombre_archivo = 'clasificador_iris_pipeline.joblib'
joblib.dump(modelo_pipeline, nombre_archivo)
print(f"\nModelo de pipeline guardado exitosamente como '{nombre_archivo}'")
`;

const codeFastAPI = `# Archivo: main.py

# --- Importaciones Necesarias ---
import uvicorn
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Creación de la Aplicación y Carga del Modelo ---
app = FastAPI(
    title="API de Clasificación de Flores Iris",
    description="Una API que utiliza un modelo de ML para predecir la especie de una flor Iris.",
    version="1.0"
)

try:
    modelo = joblib.load('clasificador_iris_pipeline.joblib')
    nombres_especies = ['setosa', 'versicolor', 'virginica']
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print("Error: Archivo 'clasificador_iris_pipeline.joblib' no encontrado.")
    modelo = None
    nombres_especies = []

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

@app.post("/predict")
def predict(request: IrisRequest):
    if modelo is None:
        return {"error": "El modelo no está disponible. Revisa los logs del servidor."}
    data_para_predecir = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])
    prediccion_numerica = modelo.predict(data_para_predecir)
    probabilidades = modelo.predict_proba(data_para_predecir)
    confianza = float(np.max(probabilidades))
    nombre_predicho = nombres_especies[prediccion_numerica[0]]
    return {
        "prediction": nombre_predicho,
        "confidence": round(confianza, 4)
    }

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Clasificación de Iris. Ve a /docs para la documentación."}
`;

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

const ModeloAServicio = ({ isDarkTheme, onToggleTheme }) => (
  <div className="admin-panel">
    <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <div className="main-content" style={{ marginLeft: 260, maxWidth: '100%', background: 'transparent', minHeight: '100vh', paddingBottom: 60 }}>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="gear">🛠️</span> Del Modelo al Servicio - Pipelines y APIs
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Introducción: ¿Por qué ir más allá de un script?</b><br/>
          En un entorno real, no basta con tener un modelo que funciona en un Jupyter Notebook. Necesitamos un sistema que sea:<br/>
          <ul>
            <li><b>Reproducible:</b> Cualquiera en tu equipo debe poder ejecutar tu código y obtener los mismos resultados.</li>
            <li><b>Robusto:</b> El sistema debe manejar los datos de manera consistente, desde el entrenamiento hasta la predicción.</li>
            <li><b>Desplegable:</b> El modelo final debe ser accesible para otros sistemas o usuarios a través de una interfaz, como una API.</li>
          </ul>
          Aquí es donde entran en juego los <b>Pipelines</b> y los <b>Servicios API</b>.<br/><br/>
          <b>1. Diseño de un Pipeline de Entrenamiento y Evaluación</b><br/>
          Un pipeline en Scikit-learn es un objeto que encapsula una secuencia de pasos de transformación y modelado. En lugar de aplicar cada paso manualmente (escalar, luego entrenar), los encadenas en un único objeto.<br/>
          <ul>
            <li><b>Simplicidad y Organización:</b> Condensa todo tu flujo de trabajo de preprocesamiento y modelado en un solo objeto.</li>
            <li><b>Prevención de Fuga de Datos (Data Leakage):</b> El pipeline asegura que los pasos de preprocesamiento se "ajusten" solo con los datos de entrenamiento y luego se apliquen a los de prueba.</li>
            <li><b>Facilidad de Uso:</b> Una vez construido, el pipeline se comporta como un modelo normal. Puedes usar <code>.fit()</code>, <code>.predict()</code> y <code>.score()</code> directamente sobre él.</li>
          </ul>
          <div style={subtitle}><span>🧑‍💻</span>Ejemplo Práctico: Construyendo un Pipeline</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePipeline}</SyntaxHighlighter>
          <b>2. Creación de un Servicio API para Predicciones</b><br/>
          Un API permite que diferentes aplicaciones se comuniquen entre sí. Crearemos un servicio web que exponga un endpoint, reciba datos de una flor, cargue el pipeline y devuelva la predicción.<br/>
          <div style={subtitle}><span>🧑‍💻</span>Ejemplo de API con FastAPI</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeFastAPI}</SyntaxHighlighter>
          <div style={{ margin: '18px 0' }}>
            <b>¿Cómo Ejecutar y Probar tu API?</b><br/>
            <ol style={{ margin: '10px 0 0 20px' }}>
              <li>Asegúrate de tener los archivos <code>main.py</code> y <code>clasificador_iris_pipeline.joblib</code> en la misma carpeta.</li>
              <li>Ejecuta el servidor con: <code>uvicorn main:app --reload</code></li>
              <li>Abre <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer">http://127.0.0.1:8000/docs</a> para probar tu API con Swagger UI.</li>
            </ol>
            ¡Has desplegado exitosamente tu modelo de machine learning como un servicio profesional!
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

export default ModeloAServicio; 