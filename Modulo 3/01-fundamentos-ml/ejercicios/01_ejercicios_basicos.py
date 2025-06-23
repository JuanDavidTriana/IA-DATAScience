# Ejercicios Básicos de Machine Learning
# =====================================

# Objetivos:
# - Practicar regresión lineal con datos reales
# - Implementar regresión logística para clasificación
# - Evaluar modelos usando diferentes métricas
# - Interpretar resultados y visualizaciones

# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_boston, load_iris
from sklearn.preprocessing import StandardScaler

# Configuración
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("✅ Librerías importadas correctamente")

# Ejercicio 1: Regresión Lineal con Datos de Boston Housing
# =========================================================

# Cargar el dataset de Boston Housing
boston = load_boston()
X_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
y_boston = boston.target

print("📊 DATASET BOSTON HOUSING")
print("=" * 40)
print(f"Forma del dataset: {X_boston.shape}")
print(f"Número de características: {X_boston.shape[1]}")
print(f"Número de muestras: {X_boston.shape[0]}")
print(f"\nCaracterísticas disponibles:")
for i, feature in enumerate(boston.feature_names):
    print(f"  {i+1:2d}. {feature}")
print(f"\nVariable objetivo: Precio de vivienda en miles de dólares")

# Exploración básica de datos
print("\n📈 ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 40)
print(X_boston.describe())

print("\n🎯 VARIABLE OBJETIVO")
print("=" * 40)
print(f"Media: {y_boston.mean():.2f}")
print(f"Desviación estándar: {y_boston.std():.2f}")
print(f"Mínimo: {y_boston.min():.2f}")
print(f"Máximo: {y_boston.max():.2f}")

# Visualizar la distribución de la variable objetivo
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(y_boston, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Precio de Vivienda (miles de $)')
plt.ylabel('Frecuencia')
plt.title('Distribución de Precios')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.boxplot(y_boston)
plt.ylabel('Precio de Vivienda (miles de $)')
plt.title('Boxplot de Precios')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.scatter(X_boston['RM'], y_boston, alpha=0.6)
plt.xlabel('Número de Habitaciones (RM)')
plt.ylabel('Precio de Vivienda (miles de $)')
plt.title('Precio vs Habitaciones')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Matriz de correlación
correlation_matrix = X_boston.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación - Características')
plt.tight_layout()
plt.show()

# TAREA 1: Implementar Regresión Lineal
# -------------------------------------
# Instrucciones:
# 1. Selecciona las 3 características más correlacionadas con el precio
# 2. Divide los datos en entrenamiento (80%) y test (20%)
# 3. Entrena un modelo de regresión lineal
# 4. Evalúa el modelo usando MSE, RMSE y R²
# 5. Visualiza los resultados

# 1. Seleccionar características más correlacionadas
correlation_with_target = X_boston.corrwith(pd.Series(y_boston)).abs().sort_values(ascending=False)
print("Correlación con el precio:")
print(correlation_with_target.head())

# Seleccionar las 3 características más correlacionadas
top_features = correlation_with_target.head(3).index.tolist()
print(f"\nCaracterísticas seleccionadas: {top_features}")

X_selected = X_boston[top_features]

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_boston, 
                                                    test_size=0.2, random_state=42)

print(f"\nForma de datos de entrenamiento: {X_train.shape}")
print(f"Forma de datos de test: {X_test.shape}")

# 3. Entrenar modelo
modelo_boston = LinearRegression()
modelo_boston.fit(X_train, y_train)

# 4. Hacer predicciones
y_pred_train = modelo_boston.predict(X_train)
y_pred_test = modelo_boston.predict(X_test)

print("✅ Modelo entrenado correctamente")
print(f"\nCoeficientes:")
for feature, coef in zip(top_features, modelo_boston.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"Intercepto: {modelo_boston.intercept_:.4f}")

# 5. Evaluar modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n📊 EVALUACIÓN DEL MODELO")
print("=" * 40)
print(f"MSE (Entrenamiento): {mse_train:.4f}")
print(f"MSE (Test): {mse_test:.4f}")
print(f"RMSE (Test): {np.sqrt(mse_test):.4f}")
print(f"R² (Entrenamiento): {r2_train:.4f}")
print(f"R² (Test): {r2_test:.4f}")

# 6. Visualizar resultados
plt.figure(figsize=(15, 5))

# Predicciones vs Valores reales
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales (Test)')
plt.grid(True, alpha=0.3)

# Residuos
residuos = y_test - y_pred_test
plt.subplot(1, 3, 2)
plt.scatter(y_pred_test, residuos, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicciones')
plt.ylabel('Residuos')
plt.title('Análisis de Residuos')
plt.grid(True, alpha=0.3)

# Distribución de residuos
plt.subplot(1, 3, 3)
plt.hist(residuos, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.title('Distribución de Residuos')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Ejercicio 2: Clasificación con Regresión Logística
# ==================================================

# Cargar dataset de Iris
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

print("\n🌸 DATASET IRIS")
print("=" * 30)
print(f"Forma del dataset: {X_iris.shape}")
print(f"Características: {iris.feature_names}")
print(f"Clases: {iris.target_names}")
print(f"Distribución de clases:")
for i, class_name in enumerate(iris.target_names):
    count = (y_iris == i).sum()
    print(f"  {class_name}: {count} muestras")

# Visualizar el dataset de Iris
plt.figure(figsize=(15, 10))

# Scatter plot de las dos primeras características
plt.subplot(2, 2, 1)
for i, class_name in enumerate(iris.target_names):
    mask = y_iris == i
    plt.scatter(X_iris.iloc[mask, 0], X_iris.iloc[mask, 1], 
               label=class_name, alpha=0.7, s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris: Características 1 vs 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Scatter plot de las dos últimas características
plt.subplot(2, 2, 2)
for i, class_name in enumerate(iris.target_names):
    mask = y_iris == i
    plt.scatter(X_iris.iloc[mask, 2], X_iris.iloc[mask, 3], 
               label=class_name, alpha=0.7, s=50)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title('Iris: Características 3 vs 4')
plt.legend()
plt.grid(True, alpha=0.3)

# Boxplot de características
plt.subplot(2, 2, 3)
X_iris.boxplot()
plt.title('Distribución de Características')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Matriz de correlación
plt.subplot(2, 2, 4)
correlation_iris = X_iris.corr()
sns.heatmap(correlation_iris, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación')

plt.tight_layout()
plt.show()

# TAREA 2: Clasificación Binaria
# ------------------------------
# Instrucciones:
# 1. Crear un problema de clasificación binaria (Setosa vs No-Setosa)
# 2. Usar solo 2 características (longitud y ancho del pétalo)
# 3. Dividir datos en entrenamiento (80%) y test (20%)
# 4. Entrenar regresión logística
# 5. Evaluar con accuracy, precision, recall y F1-score
# 6. Visualizar la frontera de decisión

# 1. Crear problema binario (Setosa vs No-Setosa)
y_binary = (y_iris == 0).astype(int)  # 0: No-Setosa, 1: Setosa

# 2. Seleccionar características (pétalo)
X_binary = X_iris[['petal length (cm)', 'petal width (cm)']]

print("\n🌺 CLASIFICACIÓN BINARIA: SETOSA vs NO-SETOSA")
print("=" * 50)
print(f"Forma de X: {X_binary.shape}")
print(f"Características: {X_binary.columns.tolist()}")
print(f"Distribución de clases:")
print(f"  No-Setosa (0): {(y_binary == 0).sum()} muestras")
print(f"  Setosa (1): {(y_binary == 1).sum()} muestras")

# 3. Dividir datos
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nDatos de entrenamiento: {X_train_bin.shape}")
print(f"Datos de test: {X_test_bin.shape}")
print(f"\nDistribución en entrenamiento:")
print(f"  No-Setosa: {(y_train_bin == 0).sum()}")
print(f"  Setosa: {(y_train_bin == 1).sum()}")

# 4. Entrenar modelo
modelo_iris = LogisticRegression(random_state=42)
modelo_iris.fit(X_train_bin, y_train_bin)

# Predicciones
y_pred_train_bin = modelo_iris.predict(X_train_bin)
y_pred_test_bin = modelo_iris.predict(X_test_bin)

print("\n✅ Modelo de clasificación entrenado")
print(f"\nCoeficientes:")
for feature, coef in zip(X_binary.columns, modelo_iris.coef_[0]):
    print(f"  {feature}: {coef:.4f}")
print(f"Intercepto: {modelo_iris.intercept_[0]:.4f}")

# 5. Evaluar modelo
accuracy_train = accuracy_score(y_train_bin, y_pred_train_bin)
accuracy_test = accuracy_score(y_test_bin, y_pred_test_bin)

print("\n📊 EVALUACIÓN DEL MODELO DE CLASIFICACIÓN")
print("=" * 50)
print(f"Accuracy (Entrenamiento): {accuracy_train:.4f}")
print(f"Accuracy (Test): {accuracy_test:.4f}")
print("\n📋 REPORTE DETALLADO (Test):")
print(classification_report(y_test_bin, y_pred_test_bin, 
                          target_names=['No-Setosa', 'Setosa']))

# 6. Visualizar frontera de decisión
def plot_decision_boundary_binary(X, y, model, title):
    # Crear malla de puntos
    x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
    y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Predecir para todos los puntos
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Graficar
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, 
                         alpha=0.8, s=50, cmap='RdYlBu', edgecolors='black')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plot_decision_boundary_binary(X_train_bin, y_train_bin, modelo_iris, 
                             'Frontera de Decisión - Entrenamiento')

plt.subplot(1, 3, 2)
plot_decision_boundary_binary(X_test_bin, y_test_bin, modelo_iris, 
                             'Frontera de Decisión - Test')

plt.subplot(1, 3, 3)
# Matriz de confusión
cm = confusion_matrix(y_test_bin, y_pred_test_bin)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No-Setosa', 'Setosa'],
            yticklabels=['No-Setosa', 'Setosa'])
plt.title('Matriz de Confusión')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')

plt.tight_layout()
plt.show()

# Ejercicio 3: Desafío Personalizado
# ===================================

# Crea tu propio dataset sintético y aplica lo aprendido:
# 1. Genera datos sintéticos para regresión lineal
# 2. Genera datos sintéticos para clasificación
# 3. Aplica los modelos correspondientes
# 4. Evalúa y visualiza los resultados
# 5. Compara el rendimiento entre diferentes configuraciones

from sklearn.datasets import make_regression, make_classification

# Generar datos de regresión
X_synth_reg, y_synth_reg = make_regression(n_samples=200, n_features=2, 
                                          noise=15, random_state=42)

print("\n📊 DATOS SINTÉTICOS - REGRESIÓN")
print("=" * 40)
print(f"Forma: {X_synth_reg.shape}")
print(f"Rango de valores objetivo: [{y_synth_reg.min():.2f}, {y_synth_reg.max():.2f}]")

# Generar datos de clasificación
X_synth_clf, y_synth_clf = make_classification(n_samples=200, n_features=2, 
                                              n_redundant=0, n_informative=2,
                                              random_state=42, n_clusters_per_class=1)

print("\n🌺 DATOS SINTÉTICOS - CLASIFICACIÓN")
print("=" * 40)
print(f"Forma: {X_synth_clf.shape}")
print(f"Clases únicas: {np.unique(y_synth_clf)}")
print(f"Distribución: {np.bincount(y_synth_clf)}")

# Visualizar datos sintéticos
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_synth_reg[:, 0], X_synth_reg[:, 1], c=y_synth_reg, 
                     cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Datos Sintéticos - Regresión')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_synth_clf[:, 0], X_synth_clf[:, 1], c=y_synth_clf, 
                     cmap='Set1', alpha=0.7, s=50)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Datos Sintéticos - Clasificación')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Aplicar modelos a datos sintéticos
# Regresión
X_train_synth, X_test_synth, y_train_synth, y_test_synth = train_test_split(
    X_synth_reg, y_synth_reg, test_size=0.2, random_state=42
)

modelo_synth_reg = LinearRegression()
modelo_synth_reg.fit(X_train_synth, y_train_synth)
y_pred_synth = modelo_synth_reg.predict(X_test_synth)

print("\n📊 RESULTADOS DATOS SINTÉTICOS")
print("=" * 40)
print(f"R² Score: {r2_score(y_test_synth, y_pred_synth):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_synth, y_pred_synth)):.4f}")

print("\n🎓 RESUMEN Y CONCLUSIONES")
print("=" * 50)
print("✅ Lo que hemos aprendido:")
print("  - Regresión Lineal: Predice valores continuos")
print("  - Regresión Logística: Clasificación binaria")
print("  - Flujo de Trabajo: Datos → Train/Test → Entrenar → Evaluar")
print("  - Métricas: MSE/RMSE/R² para regresión, Accuracy/Precision/Recall para clasificación")
print("\n🎯 Próximos Pasos:")
print("  - Regularización (Ridge, Lasso)")
print("  - Árboles de decisión")
print("  - Algoritmos más avanzados")
print("  - Validación cruzada")
print("  - Ajuste de hiperparámetros") 