# Fundamentos de Machine Learning

Esta sección contiene el material introductorio a Machine Learning con ejemplos prácticos y ejercicios.

## 📁 Estructura

```
01-fundamentos-ml/
├── teoria/
│   └── 01_introduccion_ml.ipynb    # Introducción teórica con ejemplos
├── ejercicios/
│   └── 01_ejercicios_basicos.ipynb # Ejercicios prácticos
├── requirements.txt                 # Dependencias necesarias
└── README.md                       # Este archivo
```

## 🎯 Objetivos de Aprendizaje

### Teoría (01_introduccion_ml.ipynb)
- ✅ Conceptos básicos de Machine Learning
- ✅ Tipos de aprendizaje (supervisado, no supervisado, refuerzo)
- ✅ Regresión lineal: teoría y práctica
- ✅ Regresión logística: clasificación binaria
- ✅ Métricas de evaluación
- ✅ Visualizaciones y análisis de datos

### Ejercicios (01_ejercicios_basicos.ipynb)
- ✅ **Ejercicio 1**: Regresión lineal con Boston Housing
- ✅ **Ejercicio 2**: Clasificación con dataset Iris
- ✅ **Ejercicio 3**: Datos sintéticos personalizados
- ✅ Exploración de datos y visualización
- ✅ Evaluación de modelos
- ✅ Interpretación de resultados

## 🚀 Cómo Empezar

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar Jupyter
```bash
jupyter notebook
```

### 3. Orden de Estudio
1. **Teoría**: `teoria/01_introduccion_ml.ipynb`
2. **Ejercicios**: `ejercicios/01_ejercicios_basicos.ipynb`

## 📊 Datasets Utilizados

### Boston Housing Dataset
- **Propósito**: Regresión (predecir precios de viviendas)
- **Características**: 13 variables (habitaciones, crimen, etc.)
- **Objetivo**: Precio de vivienda en miles de dólares

### Iris Dataset
- **Propósito**: Clasificación (identificar especies de iris)
- **Características**: 4 variables (longitud/ancho de pétalo/sépalo)
- **Objetivo**: 3 clases (Setosa, Versicolor, Virginica)

## 🎓 Conceptos Clave

### Regresión Lineal
- **Función**: y = β₀ + β₁x + ε
- **Métricas**: MSE, RMSE, R²
- **Aplicaciones**: Predicción de precios, ventas, etc.

### Regresión Logística
- **Función**: P(y=1) = 1 / (1 + e^(-z))
- **Métricas**: Accuracy, Precision, Recall, F1-Score
- **Aplicaciones**: Clasificación binaria, spam detection

### Flujo de Trabajo
1. 📊 **Exploración de datos**
2. 🔄 **División train/test**
3. 🎯 **Entrenamiento del modelo**
4. 📈 **Evaluación**
5. 🔍 **Interpretación**

## 💡 Tips de Aprendizaje

- **Ejecuta todas las celdas** en orden
- **Experimenta** cambiando parámetros
- **Visualiza** los resultados
- **Interpreta** las métricas
- **Practica** con tus propios datos

## 🔗 Recursos Adicionales

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Gallery](https://matplotlib.org/gallery/)
- [Seaborn Examples](https://seaborn.pydata.org/examples/)

## 📝 Notas

- Los notebooks incluyen explicaciones detalladas
- Cada ejercicio tiene instrucciones claras
- Se incluyen visualizaciones para mejor comprensión
- Los códigos están comentados para facilitar el aprendizaje 