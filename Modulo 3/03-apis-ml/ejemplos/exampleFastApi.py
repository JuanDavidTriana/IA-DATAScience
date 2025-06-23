# Archivo: main.py (Versión Robusta)

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