from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="API de Predicción de Precios de Viviendas USA",
    description="API para predecir precios de viviendas usando modelo de regresión lineal",
    version="1.0.0"
)

class ViviendaInput(BaseModel):
    avg_area_income: float
    avg_area_house_age: float
    avg_area_number_of_rooms: float
    avg_area_number_of_bedrooms: float
    area_population: float

class PrediccionResponse(BaseModel):
    precio_predicho: float
    precio_formateado: str
    confianza: str

modelo = None # Variable global para almacenar el modelo

def cargar_modelo():
    global modelo
    try:
        with open('modelo_regresion_lineal.pkl', 'rb') as file:
            modelo = pickle.load(file)
        print('Modelo cargado')
        return True
    except FileNotFoundError:
        print('Modelo no encontrado')
        return False
    except Exception as e:
        raise HTTPException(status_code=500, detail='Error al cargar el modelo')
    
@app.on_event('startup')
async def startup_event():
    print('Iniciando API de Prediccion de Vivienda en USA')
    if not cargar_modelo():
        raise HTTPException(status_code=500, detail='Error al cargar el modelo')
    
@app.get('/')
async def index():
    return {'message': 'API de Prediccion de Vivienda en USA',
            'version': '1.0.0',
            'endpoint': {
                '/': "Informacion de la API",
                '/predict': "Prediccion de Precio de Vivienda en USA",
                '/predict-batch': "Hacer multiples predicciones de Precio de Vivienda en USA",
                "/health": "Estado de la API",
                "/docs": "Documentacion de la API"
            }}

@app.get('/health')
async def health_check():
    modelo_cargado = modelo is not None
    return {
            'status': 'OK' if modelo_cargado else 'Error',
            'modelo_cargado': modelo_cargado,
            'mensaje': 'API funcionando correctamente' if modelo_cargado else 'Modelo no disponible'
            }

@app.post('/predict', response_model=PrediccionResponse)
async def predecir_precio(vivienda: ViviendaInput):

    if modelo is None:
        raise HTTPException(status_code=500, detail='Modelo no disponible')

    try:
        datos = pd.DataFrame({
            'Avg. Area Income': [vivienda.avg_area_income],
            'Avg. Area House Age': [vivienda.avg_area_house_age],
            'Avg. Area Number of Rooms': [vivienda.avg_area_number_of_rooms],
            'Avg. Area Number of Bedrooms': [vivienda.avg_area_number_of_bedrooms],
            'Area Population': [vivienda.area_population]
        })

        prediccion = modelo.predict(datos)[0]

        if prediccion < 500000:
            confianza = "Baja"
        elif prediccion < 1000000:
            confianza = "Media"
        elif prediccion < 1500000:
            confianza = "Alta"
        else:
            confianza = "Muy Alta"

        return PrediccionResponse(
            precio_predicho=prediccion,
            precio_formateado=f'${prediccion:,.2f}',
            confianza=confianza
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir el precio: {str(e)}")
        
@app.post("/predict-batch")
async def predicir_precios_batch(viviendas: List[ViviendaInput]):

    if modelo is None:
        raise HTTPException(status_code=500, detail='Modelo no disponible')

    try:
        resultados = []

        for i, vivienda in enumerate(viviendas):
            datos = pd.DataFrame({
                'Avg. Area Income': [vivienda.avg_area_income],
                'Avg. Area House Age': [vivienda.avg_area_house_age],
                'Avg. Area Number of Rooms': [vivienda.avg_area_number_of_rooms],
                'Avg. Area Number of Bedrooms': [vivienda.avg_area_number_of_bedrooms],
                'Area Population': [vivienda.area_population]
            })

            prediccion = modelo.predict(datos)[0]

            resultados.append({
                'vivienda_id': i + 1,
                'precio_predicho': prediccion,
                'precio_formateado': f'${prediccion:,.2f}',
                "datos_entrada": {
                    "Avg. Area Income": vivienda.avg_area_income,
                    "Avg. Area House Age": vivienda.avg_area_house_age,
                    "Avg. Area Number of Rooms": vivienda.avg_area_number_of_rooms,
                    "Avg. Area Number of Bedrooms": vivienda.avg_area_number_of_bedrooms,
                    "Area Population": vivienda.area_population
                }
            })

        return {
            "total_predicciones": len(viviendas),
            "predicciones": resultados
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir el precio: {str(e)}")

@app.get("/ejemplo")
async def obtener_ejemplo_datos():
    return {
        "ejemplo_datos": {
            "Avg. Area Income": 8.3252,
            "Avg. Area House Age": 41.0,
            "Avg. Area Number of Rooms": 6.984127,
            "Avg. Area Number of Bedrooms": 1.02381,
            "Area Population": 322.0
        },
        'descripcion': 'Ejemplo de datos para predecir precio de vivienda en USA',
        'uso': 'Puedes usar estos datos para predecir el precio de una vivienda en USA'
    }

if __name__ == '__main__':
    print('Iniciando API de Prediccion de Vivienda en USA')
    uvicorn.run(app, host='0.0.0.0', port=8000)