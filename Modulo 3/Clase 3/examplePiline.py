# -*- coding: utf-8 -*-

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
print("¡Entrenamiento completado!")

# --- 4. Evaluar el Pipeline ---
y_pred = modelo_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"La exactitud del pipeline en el conjunto de prueba es: {accuracy:.2%}")

# --- 5. Guardar el Pipeline Entrenado para Producción ---
nombre_archivo = 'clasificador_iris_pipeline.joblib'
joblib.dump(modelo_pipeline, nombre_archivo)
print(f"Modelo de pipeline guardado exitosamente como '{nombre_archivo}'")