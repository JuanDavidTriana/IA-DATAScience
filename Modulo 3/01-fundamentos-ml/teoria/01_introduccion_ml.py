# Fundamentos de Machine Learning
# ===============================

# ¿Qué es Machine Learning?
# Machine Learning (ML) es una rama de la Inteligencia Artificial que permite 
# a las computadoras aprender y mejorar automáticamente a partir de la experiencia 
# sin ser programadas explícitamente.

# Tipos de Machine Learning:
# 1. Aprendizaje Supervisado: El algoritmo aprende de datos etiquetados
# 2. Aprendizaje No Supervisado: El algoritmo encuentra patrones en datos sin etiquetar
# 3. Aprendizaje por Refuerzo: El algoritmo aprende interactuando con un entorno

# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification

# Configuración para mejor visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Librerías importadas correctamente")

# Ejemplo 1: Regresión Lineal Simple
# ==================================

# Generar datos sintéticos para regresión lineal
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Dividir en entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
print(f"Forma de y_test: {y_test.shape}")

# Visualizar los datos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.7, label='Datos de Entrenamiento')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Datos de Entrenamiento')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.7, color='red', label='Datos de Test')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Datos de Test')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Crear y entrenar el modelo de regresión lineal
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# Hacer predicciones
y_pred_train = modelo_regresion.predict(X_train)
y_pred_test = modelo_regresion.predict(X_test)

print("✅ Modelo entrenado correctamente")
print(f"Coeficiente (pendiente): {modelo_regresion.coef_[0]:.4f}")
print(f"Intercepto: {modelo_regresion.intercept_:.4f}")

# Visualizar el modelo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.7, label='Datos Reales')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Predicción')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Entrenamiento: Datos vs Predicción')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.7, label='Datos Reales')
plt.plot(X_test, y_pred_test, color='red', linewidth=2, label='Predicción')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Test: Datos vs Predicción')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluar el modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("📊 MÉTRICAS DE EVALUACIÓN")
print("=" * 40)
print(f"MSE (Entrenamiento): {mse_train:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"R² (Entrenamiento): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")
print(f"RMSE (Test): {np.sqrt(mse_test):.4f}")

# Ejemplo 2: Regresión Logística (Clasificación)
# ==============================================

# Generar datos sintéticos para clasificación
X_clf, y_clf = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                   n_informative=2, random_state=42, n_clusters_per_class=1)

# Dividir en entrenamiento y test
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

print(f"Forma de X_train_clf: {X_train_clf.shape}")
print(f"Forma de y_train_clf: {y_train_clf.shape}")
print(f"Clases únicas: {np.unique(y_clf)}")

# Visualizar los datos de clasificación
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_train_clf[:, 0], X_train_clf[:, 1], c=y_train_clf, 
                     cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Datos de Entrenamiento (Clasificación)')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_test_clf[:, 0], X_test_clf[:, 1], c=y_test_clf, 
                     cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Datos de Test (Clasificación)')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Crear y entrenar el modelo de regresión logística
modelo_clasificacion = LogisticRegression(random_state=42)
modelo_clasificacion.fit(X_train_clf, y_train_clf)

# Hacer predicciones
y_pred_train_clf = modelo_clasificacion.predict(X_train_clf)
y_pred_test_clf = modelo_clasificacion.predict(X_test_clf)

print("✅ Modelo de clasificación entrenado correctamente")
print(f"Coeficientes: {modelo_clasificacion.coef_[0]}")
print(f"Intercepto: {modelo_clasificacion.intercept_[0]:.4f}")

# Evaluar el modelo de clasificación
accuracy_train = accuracy_score(y_train_clf, y_pred_train_clf)
accuracy_test = accuracy_score(y_test_clf, y_pred_test_clf)

print("📊 MÉTRICAS DE CLASIFICACIÓN")
print("=" * 40)
print(f"Accuracy (Entrenamiento): {accuracy_train:.4f}")
print(f"Accuracy (Test): {accuracy_test:.4f}")
print("\n📋 REPORTE DE CLASIFICACIÓN (Test):")
print(classification_report(y_test_clf, y_pred_test_clf))

# Visualizar la frontera de decisión
def plot_decision_boundary(X, y, model, title):
    # Crear una malla de puntos
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predecir para todos los puntos de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, s=50, cmap='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_boundary(X_train_clf, y_train_clf, modelo_clasificacion, 
                      'Frontera de Decisión - Entrenamiento')

plt.subplot(1, 2, 2)
plot_decision_boundary(X_test_clf, y_test_clf, modelo_clasificacion, 
                      'Frontera de Decisión - Test')

plt.tight_layout()
plt.show()

print("\n🎓 RESUMEN DE CONCEPTOS CLAVE")
print("=" * 50)
print("✅ Regresión Lineal: Predice valores continuos")
print("✅ Regresión Logística: Predice clases (0 o 1)")
print("✅ Flujo de Trabajo: Datos → Train/Test → Entrenar → Evaluar")
print("✅ Métricas: MSE/RMSE/R² para regresión, Accuracy/Precision/Recall para clasificación") 