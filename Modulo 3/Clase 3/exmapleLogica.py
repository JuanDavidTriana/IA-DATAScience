# -*- coding: utf-8 -*-

# 1. Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# 2. Caso de Uso: Riesgo Crediticio
np.random.seed(42)
X1 = np.random.normal(35, 10, 200); X2 = np.random.normal(50000, 15000, 200)

X = np.vstack((X1, X2)).T
probabilidad_default = 1 / (1 + np.exp(-( (X[:,0] - 40)/5 + (60000 - X[:,1])/20000 )))
y = (probabilidad_default > np.random.rand(200)).astype(int)

# 3. Preprocesamiento: Escalar Características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 5. Crear y Entrenar el Modelo
modelo_logistico = LogisticRegression()
modelo_logistico.fit(X_train, y_train)

# 6. Evaluar el Rendimiento
y_pred = modelo_logistico.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud (Accuracy): {accuracy:.2f}")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 7. Usar el modelo para una decisión de negocio
nuevo_cliente = np.array([[38, 80000]])
nuevo_cliente_scaled = scaler.transform(nuevo_cliente)
prediccion_clase = modelo_logistico.predict(nuevo_cliente_scaled)
probabilidad_riesgo = modelo_logistico.predict_proba(nuevo_cliente_scaled)

print(f"--- Decisión para Nuevo Cliente: {nuevo_cliente[0]} ---")
print(f"Clase Predicha (0=Bajo Riesgo, 1=Alto Riesgo): {prediccion_clase[0]}")
print(f"Probabilidad de ser Bajo Riesgo (Clase 0): {probabilidad_riesgo[0][0]:.2%}")
print(f"Probabilidad de ser Alto Riesgo (Clase 1): {probabilidad_riesgo[0][1]:.2%}")
print("Decisión Sugerida: Aprobar el préstamo, el riesgo de default es bajo.")