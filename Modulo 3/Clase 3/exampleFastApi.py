# Archivo: main.py

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