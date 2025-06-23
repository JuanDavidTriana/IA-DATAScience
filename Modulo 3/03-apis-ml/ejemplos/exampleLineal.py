# -*- coding: utf-8 -*-

# 1. Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score # Importar métricas de evaluación

# 2. Caso de Uso: Marketing
# Se quiere predecir las ventas (y) a partir de la inversión en publicidad por TV (X).
np.random.seed(0)
X = np.random.rand(100, 1) * 100
y = 50 + 2 * X.flatten() + np.random.randn(100) * 20

# 3. Dividir datos para simular un escenario real
# Entrenamos con datos históricos (80%) y validamos con datos nuevos (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Crear y Entrenar el Modelo
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)

print(f"Intercepto (β₀): {modelo_lineal.intercept_:.2f}")
print(f"Coeficiente (β₁): {modelo_lineal.coef_[0]:.2f}")

# 5. Realizar y Evaluar Predicciones
y_pred = modelo_lineal.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")

# 6. Visualización e Interpretación de Resultados
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Ventas Reales (Datos de Prueba)')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicciones del Modelo')
plt.title('Impacto de la Inversión en Publicidad sobre las Ventas')
plt.xlabel('Inversión en Publicidad (miles USD)')
plt.ylabel('Ventas (miles USD)')
plt.legend()
plt.grid(True)
plt.show()
