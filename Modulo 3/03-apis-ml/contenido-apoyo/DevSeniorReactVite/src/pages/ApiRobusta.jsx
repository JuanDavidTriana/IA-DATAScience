import React from 'react';
import Slidebar from '../components/Slidebar';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const codeValidacion = `from pydantic import BaseModel, Field

class IrisRequest(BaseModel):
    # Una flor no puede tener una medida negativa o cero.
    # gt=0 significa "greater than 0".
    sepal_length: float = Field(..., gt=0, description="Largo del sépalo en cm")
    sepal_width: float  = Field(..., gt=0, description="Ancho del sépalo en cm")
    petal_length: float = Field(..., gt=0, description="Largo del pétalo en cm")
    petal_width: float  = Field(..., gt=0, description="Ancho del pétalo en cm")
`;

const codeErrores = `from fastapi import FastAPI, HTTPException

app = FastAPI()

# Ejemplo de endpoint
@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id > 100:
        # Lanza una excepción con un código de estado y un mensaje claro.
        raise HTTPException(status_code=404, detail="El item no fue encontrado")
    return {"item_id": item_id}
`;

const codeLogging = `import logging

# Configuración básica del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Uso del logger
logging.info("La aplicación se está iniciando.")
try:
    # Alguna operación que podría fallar
    resultado = 10 / 0
except Exception as e:
    logging.error(f"Ocurrió un error inesperado: {e}", exc_info=True)
`;

const codeSolucion = `# Archivo: main.py (Versión Robusta)

# --- Importaciones ---
import uvicorn
import joblib
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- 1. Configuración de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 2. Definición del Modelo de Datos de Entrada con Validación Avanzada ---
class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Largo del sépalo en cm. Debe ser mayor que 0.")
    sepal_width: float  = Field(..., gt=0, description="Ancho del sépalo en cm. Debe ser mayor que 0.")
    petal_length: float = Field(..., gt=0, description="Largo del pétalo en cm. Debe ser mayor que 0.")
    petal_width: float  = Field(..., gt=0, description="Ancho del pétalo en cm. Debe ser mayor que 0.")

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1, "sepal_width": 3.5,
                "petal_length": 1.4, "petal_width": 0.2
            }
        }

# --- 3. Creación de la Aplicación y Carga del Modelo ---
app = FastAPI(
    title="API Robusta de Clasificación de Flores Iris",
    description="Una API de producción que utiliza un modelo de ML para predecir la especie de una flor Iris, con validación, manejo de errores y logging.",
    version="2.0"
)

# Carga del modelo con manejo de errores
try:
    modelo = joblib.load('clasificador_iris_pipeline.joblib')
    nombres_especies = ['setosa', 'versicolor', 'virginica']
    logging.info("Modelo de ML y nombres de especies cargados exitosamente.")
except FileNotFoundError:
    logging.error("Archivo del modelo 'clasificador_iris_pipeline.joblib' no encontrado.")
    modelo = None
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    modelo = None

# --- 4. Creación del Endpoint de Predicción Robusto ---
@app.post("/predict")
def predict(request: IrisRequest):
    """
    Recibe las medidas validadas de una flor, realiza una predicción
    y devuelve la especie junto con la confianza del modelo.
    Maneja errores de forma explícita.
    """
    # Verificación de la disponibilidad del modelo
    if modelo is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible: El modelo de ML no pudo ser cargado.")

    try:
        logging.info(f"Petición de predicción recibida: {request.dict()}")

        # Convertir datos de entrada para la predicción
        data_para_predecir = np.array([[
            request.sepal_length, request.sepal_width,
            request.petal_length, request.petal_width
        ]])

        # Realizar predicción y obtener probabilidades
        prediccion_numerica = modelo.predict(data_para_predecir)
        probabilidades = modelo.predict_proba(data_para_predecir)
        confianza = float(np.max(probabilidades))
        nombre_predicho = nombres_especies[prediccion_numerica[0]]
        
        logging.info(f"Predicción exitosa: {nombre_predicho} con confianza {confianza:.4f}")

        # Devolver el resultado
        return {
            "prediction": nombre_predicho,
            "confidence": round(confianza, 4)
        }
    except Exception as e:
        # Si algo sale mal durante la predicción, se registra y se devuelve un error 500
        logging.error(f"Error durante el proceso de predicción: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno del servidor al procesar la predicción.")

@app.get("/")
def read_root():
    """Endpoint raíz que da la bienvenida a la API."""
    return {"message": "Bienvenido a la API de Clasificación de Iris v2.0. Visita /docs para la documentación interactiva."}
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

const ApiRobusta = ({ isDarkTheme, onToggleTheme }) => (
  <div className="admin-panel">
    <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <div className="main-content" style={{ marginLeft: 260, maxWidth: '100%', background: 'transparent', minHeight: '100vh', paddingBottom: 60 }}>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="shield">🛡️</span> Del Prototipo a la Producción – APIs Robustas
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Introducción: ¿Qué es el Desarrollo Orientado a Servicios?</b><br/>
          Es un paradigma de desarrollo de software que estructura una aplicación como una colección de servicios poco acoplados y altamente mantenibles. Nuestro API de predicción es un ejemplo perfecto de un microservicio: un componente pequeño, independiente y enfocado en una sola tarea (en este caso, hacer predicciones).<br/><br/>
          Un servicio en producción no solo debe funcionar en el "caso feliz". De hecho, gran parte del código de producción se dedica a manejar lo inesperado. Aquí es donde las buenas prácticas se vuelven esenciales.<br/><br/>
          <b>1. Buenas Prácticas para Construir APIs Robustas</b>
          <div style={subtitle}>A. Validación Rigurosa de la Entrada</div>
          <i>Principio: "Basura entra, basura sale" (Garbage In, Garbage Out). Nunca confíes en los datos que envía un cliente. Valídalos siempre en el borde de tu aplicación.</i><br/><br/>
          FastAPI y Pydantic hacen esto muy fácil, pero podemos ir más allá.<br/><br/>
          <b>Cómo:</b> Usa la función <b>Field</b> de Pydantic para añadir restricciones directamente a tu modelo de datos.<br/>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeValidacion}</SyntaxHighlighter>
          Si un cliente envía <code>sepal_length: -5</code>, FastAPI automáticamente responderá con un error 422 Unprocessable Entity detallando el problema, ¡sin que tengas que escribir una sola línea de código para manejarlo!<br/><br/>
          <div style={subtitle}>B. Manejo Explícito de Errores</div>
          <i>Principio: Falla con gracia y da información útil. Un error 500 Internal Server Error sin contexto es inútil para el cliente. Debes capturar errores predecibles y devolver respuestas claras.</i><br/><br/>
          <b>Cómo:</b> Usa la excepción <b>HTTPException</b> de FastAPI.<br/>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeErrores}</SyntaxHighlighter>
          Esto le da al cliente un código de estado estándar (404 Not Found) y un mensaje JSON (<code>{'{"detail": "El item no fue encontrado"}'}</code>) que puede procesar.<br/><br/>
          <div style={subtitle}>C. Logging (Registro de Eventos)</div>
          <i>Principio: Si no está en un log, no ocurrió. En producción, no puedes simplemente poner print() en tu código. Necesitas un sistema de registro (logging) para diagnosticar problemas.</i><br/><br/>
          <b>Cómo:</b> Usa el módulo <b>logging</b> incorporado de Python.<br/>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeLogging}</SyntaxHighlighter>
          Los logs te permiten rastrear el flujo de tu aplicación, ver las solicitudes que llegan y diagnosticar errores sin detener el servicio.<br/><br/>
          <div style={subtitle}>D. Documentación Clara y Consistente</div>
          <i>Principio: Tu API es un producto para otros desarrolladores. La documentación no es un extra, es parte de la interfaz.</i><br/><br/>
          <b>Cómo:</b> Aprovecha al máximo la autogeneración de FastAPI.<br/>
          <ul>
            <li>Usa los parámetros <b>title</b> y <b>description</b> al crear la instancia de FastAPI.</li>
            <li>Añade descripciones a los modelos de Pydantic usando <b>Field</b>.</li>
            <li>Usa <b>docstrings</b> en tus funciones de endpoint para explicar lo que hacen. FastAPI los mostrará automáticamente en la documentación de <code>/docs</code>.</li>
          </ul>
          <b>2. Ejercicio Práctico: Mejora de una API para Producción</b><br/>
          <i>Objetivo:</i> Tomarás la API del clasificador de Iris de la clase anterior y la refactorizarás para que sea más robusta y esté lista para un entorno de producción, aplicando todas las buenas prácticas que acabamos de ver.<br/><br/>
          <b>Tareas a Realizar:</b>
          <ul>
            <li><b>Mejorar la Validación:</b> Modifica el modelo IrisRequest en tu archivo main.py. Utiliza Field de Pydantic para asegurar que todas las medidas de la flor sean mayores que cero (gt=0).</li>
            <li><b>Añadir Manejo de Errores:</b> En el endpoint /predict, envuelve la lógica de predicción en un bloque try...except. Si ocurre cualquier error inesperado durante la predicción, captura la excepción y lanza una HTTPException con status_code=500 y el mensaje "Error interno al procesar la predicción.".<br/>Mejora el try...except que carga el modelo al inicio. Si el archivo .joblib no se encuentra, la variable modelo será None. Modifica el endpoint /predict para que, si modelo es None, lance inmediatamente una HTTPException con status_code=503 (Service Unavailable) y el mensaje "El modelo de Machine Learning no está disponible.".</li>
            <li><b>Implementar Logging:</b> Importa y configura el módulo logging al principio de tu archivo main.py.<br/>Añade un logging.info() para registrar cuándo se carga el modelo exitosamente al inicio.<br/>Dentro del endpoint /predict, añade un logging.info() que registre los datos de la flor que se recibió. Por ejemplo: <code>logging.info(f&quot;Recibida petición de predicción: &#123;request.dict()&#125;&quot;)</code>.<br/>En tu nuevo bloque except, antes de lanzar la HTTPException 500, añade un logging.error() para registrar el error completo, incluyendo la traza de la excepción.</li>
            <li><b>Refinar la Documentación:</b> Asegúrate de que tu instancia de FastAPI tenga un title y description claros, y que tu endpoint /predict tenga un docstring explicando su función.</li>
          </ul>
          <details style={{ margin: '18px 0' }}>
            <summary style={{ cursor: 'pointer', color: '#a259f7', fontWeight: 600, fontSize: 17 }}>Haz clic aquí para ver el código de la solución refactorizada</summary>
            <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeSolucion}</SyntaxHighlighter>
          </details>
          <div style={{ marginTop: 24, fontWeight: 600, color: '#a259f7', fontSize: 18 }}>
            Al completar este ejercicio, habrás transformado un simple prototipo en un servicio robusto, aplicando prácticas que son estándar en la industria del software y MLOps.
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

export default ApiRobusta; 