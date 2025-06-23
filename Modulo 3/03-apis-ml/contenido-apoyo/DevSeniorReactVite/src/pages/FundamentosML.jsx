import React from 'react';
import Slidebar from '../components/Slidebar';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const codeLinear = `# -*- coding: utf-8 -*-

# 1. Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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

print(f"\nError Cuadrático Medio (MSE): {mse:.2f}")
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

# 7. Usar el modelo para tomar una decisión de negocio
inversion_propuesta = np.array([[85]]) # Proponemos invertir 85,000 USD
venta_estimada = modelo_lineal.predict(inversion_propuesta)
print(f"\nDecisión: Si invertimos {inversion_propuesta[0][0]} mil USD, el modelo predice ventas de {venta_estimada[0]:.2f} mil USD.")
`;

const codeLogistic = `# -*- coding: utf-8 -*-

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
print(f"Exactitud (Accuracy): {accuracy:.2f}\n")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# 7. Usar el modelo para una decisión de negocio
nuevo_cliente = np.array([[38, 80000]])
nuevo_cliente_scaled = scaler.transform(nuevo_cliente)
prediccion_clase = modelo_logistico.predict(nuevo_cliente_scaled)
probabilidad_riesgo = modelo_logistico.predict_proba(nuevo_cliente_scaled)

print(f"\n--- Decisión para Nuevo Cliente: {nuevo_cliente[0]} ---")
print(f"Clase Predicha (0=Bajo Riesgo, 1=Alto Riesgo): {prediccion_clase[0]}")
print(f"Probabilidad de ser Bajo Riesgo (Clase 0): {probabilidad_riesgo[0][0]:.2%}")
print(f"Probabilidad de ser Alto Riesgo (Clase 1): {probabilidad_riesgo[0][1]:.2%}")
print("Decisión Sugerida: Aprobar el préstamo, el riesgo de default es bajo.")
`;

const codePractica1 = `import pandas as pd

# Suponiendo que tienes un archivo CSV con tus datos
# df = pd.read_csv('ruta/a/tus/datos.csv')

# Para este ejemplo, crearemos un DataFrame de ejemplo
data = {
    'horas_estudio': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4],
    'aprobo_examen': [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1] # 1 = Sí, 0 = No
}
df = pd.DataFrame(data)

print("Primeros 5 registros de los datos:")
print(df.head())
`;

const codePractica2 = `# 'X' contiene todas las columnas de entrada (en este caso, solo una)
X = df[['horas_estudio']]

# 'y' contiene la columna de salida que queremos predecir
y = df['aprobo_examen']
`;

const codePractica3 = `from sklearn.model_selection import train_test_split

# Dividimos los datos: 80% para entrenar, 20% para probar.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
`;

const codePractica4 = `from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Se ajusta el escalador SÓLO con los datos de entrenamiento para evitar fuga de información del test set.
X_train_scaled = scaler.fit_transform(X_train)

# Se aplica la MISMA transformación a los datos de prueba.
X_test_scaled = scaler.transform(X_test)
`;

const codePractica5 = `from sklearn.linear_model import LogisticRegression

# Creamos una instancia del modelo. Podemos ajustar "hiperparámetros" aquí.
modelo = LogisticRegression(random_state=42)
`;

const codePractica6 = `# Entrenamos el modelo usando las características escaladas y las etiquetas de entrenamiento.
print("\nEntrenando el modelo...")
modelo.fit(X_train_scaled, y_train)
print("¡Modelo entrenado exitosamente!")
`;

const codePractica7 = `# El modelo ahora predice las etiquetas para el conjunto de prueba (que nunca antes había visto).
y_pred = modelo.predict(X_test_scaled)

print("\nPredicciones sobre el conjunto de prueba:")
print(f"Valores Reales (y_test): {y_test.values}")
print(f"Predicciones (y_pred):   {y_pred}")
`;

const codePractica8 = `from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy: ¿Qué porcentaje de las predicciones fue correcto?
accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitud (Accuracy) del modelo: {accuracy:.2%}")

# Matriz de Confusión: Muestra los aciertos y errores en detalle.
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Reporte de Clasificación: Ofrece un resumen completo con precisión, recall y f1-score.
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
`;

const codePracticaResumen = `# -*- coding: utf-8 -*-

# Importaciones generales
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --- PASO 1: CARGAR DATOS ---
data = {'horas_estudio': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4],
        'aprobo_examen': [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]}
df = pd.DataFrame(data)

# --- PASO 2: PREPARACIÓN Y DIVISIÓN ---
# a) Separar X e y
X = df[['horas_estudio']]
y = df['aprobo_examen']

# b) Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# c) Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PASO 3: CREAR EL MODELO ---
modelo = LogisticRegression(random_state=42)

# --- PASO 4: ENTRENAR EL MODELO ---
modelo.fit(X_train_scaled, y_train)

# --- PASO 5: REALIZAR PREDICCIONES ---
y_pred = modelo.predict(X_test_scaled)

# --- PASO 6: EVALUAR EL MODELO ---
print("--- EVALUACIÓN DEL MODELO ---")
print(f"Exactitud: {accuracy_score(y_test, y_pred):.2%}\n")
print(classification_report(y_test, y_pred))

# --- EXTRA: USAR EL MODELO EN PRODUCCIÓN ---
horas_nuevas = [[6.5]]
horas_nuevas_scaled = scaler.transform(horas_nuevas) # Usar el mismo scaler
prediccion_final = modelo.predict(horas_nuevas_scaled)
print("--- PREDICCIÓN PARA UN NUEVO DATO ---")
print(f"Un estudiante que estudia {horas_nuevas[0][0]} horas, tiene una predicción de: {'Aprobar' if prediccion_final[0] == 1 else 'No Aprobar'}")
`;

const codeIris = `# -*- coding: utf-8 -*-

# --- PASO 1: CARGAR Y EXPLORAR DATOS ---
from sklearn.datasets import load_iris

iris = load_iris()

print("--- Exploración Inicial de Datos ---")
print("Claves del dataset:", iris.keys())
print("Nombres de características:", iris.feature_names)
print("Nombres de las especies:", iris.target_names)
print("-" * 35)

# --- PASO 2: PREPARAR DATOS ---
from sklearn.model_selection import train_test_split

# a) Separar X e y
X = iris.data
y = iris.target

# b) Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Datos cargados y divididos: {X_train.shape[0]} para entrenar, {X_test.shape[0]} para probar.\n")

# --- PASO 3: CREAR EL MODELO ---
from sklearn.linear_model import LogisticRegression

# Creamos la instancia del modelo.
# max_iter se aumenta para asegurar que el algoritmo tenga suficientes iteraciones para converger.
modelo = LogisticRegression(max_iter=200, random_state=42)

# --- PASO 4: ENTRENAR EL MODELO ---
print("Entrenando el modelo de Regresión Logística...")
modelo.fit(X_train, y_train)
print("¡Entrenamiento completado!\n")

# --- PASO 5: REALIZAR PREDICCIONES ---
y_pred = modelo.predict(X_test)

# --- PASO 6: EVALUAR EL MODELO ---
from sklearn.metrics import accuracy_score, classification_report

print("--- Resultados de la Evaluación ---")
# Calculamos la exactitud
accuracy = accuracy_score(y_test, y_pred)
print(f"Exactitud (Accuracy) del modelo: {accuracy:.2%}\n")

# Imprimimos el reporte completo
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Ejemplo de predicción para una nueva flor
# Medidas: sépalo largo=5.1cm, sépalo ancho=3.5cm, pétalo largo=1.4cm, pétalo ancho=0.2cm (típicamente una 'setosa')
nueva_flor = [[5.1, 3.5, 1.4, 0.2]]
prediccion_nueva = modelo.predict(nueva_flor)
especie_predicha = iris.target_names[prediccion_nueva[0]]

print("\n--- Prueba con una nueva flor ---")
print(f"La nueva flor se clasifica como: '{especie_predicha}'")
`;

const codePipeline = `# -*- coding: utf-8 -*-

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
print("¡Entrenamiento completado!\n")

# --- 4. Evaluar el Pipeline ---
y_pred = modelo_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"La exactitud del pipeline en el conjunto de prueba es: {accuracy:.2%}")

# --- 5. Guardar el Pipeline Entrenado para Producción ---
nombre_archivo = 'clasificador_iris_pipeline.joblib'
joblib.dump(modelo_pipeline, nombre_archivo)
print(f"\nModelo de pipeline guardado exitosamente como '{nombre_archivo}'")
`;

const codeFastAPI = `# Archivo: main.py

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

const divider = {
  border: 'none',
  borderTop: '1.5px solid #e5e7eb',
  margin: '32px 0',
};

const highlight = { color: '#a259f7', fontWeight: 700 };

const irisImgUrl = "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg";
const irisDatasetImg = "https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_dataset_001.png";

const FundamentosML = ({ isDarkTheme, onToggleTheme }) => (
  <div className="admin-panel">
    <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <div className="main-content" style={{ marginLeft: 260, maxWidth: '100%', background: 'transparent', minHeight: '100vh', paddingBottom: 60 }}>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="brain">🧠</span> Fundamentos de Machine Learning
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>¿Qué es el Machine Learning?</b><br/>
          El <b>Machine Learning (ML)</b> o Aprendizaje Automático es una disciplina de la inteligencia artificial que se enfoca en el desarrollo de algoritmos que otorgan a las computadoras la habilidad de aprender de los datos. En lugar de ser programadas con reglas explícitas, las máquinas identifican patrones complejos en los datos de entrenamiento para realizar predicciones o tomar decisiones inteligentes sobre datos nunca antes vistos.
          <ul style={{ marginTop: 16, marginBottom: 0 }}>
            <li><b>Aprendizaje Supervisado:</b> El más común en la industria. Entrenas al modelo con un conjunto de datos que contiene tanto las "preguntas" (features) como las "respuestas" (labels).</li>
            <li><b>Aprendizaje No Supervisado:</b> El algoritmo explora datos sin etiquetar para descubrir patrones ocultos.</li>
            <li><b>Aprendizaje por Refuerzo:</b> Se utiliza para entrenar agentes que aprenden a tomar decisiones secuenciales interactuando con un entorno.</li>
          </ul>
        </div>
      </div>
      <hr style={divider} />
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="chart">📈</span> <span style={highlight}>Regresión Lineal: Prediciendo el Futuro Numérico</span>
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          La regresión lineal es el punto de partida por excelencia en el aprendizaje supervisado. Busca modelar una relación lineal entre una variable de salida continua (dependiente) y una o más variables de entrada (independientes).
          <div style={subtitle}><span>📚</span>Teoría Ampliada</div>
          El objetivo de la regresión lineal es encontrar la línea (o hiperplano en múltiples dimensiones) que mejor se ajuste a los datos. Esto se logra minimizando el <b style={highlight}>Error Cuadrático Medio (MSE)</b>:
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 16, margin: '18px 0' }}>
{`MSE = (1/n) * Σ(yi - ŷi)^2`}
          </SyntaxHighlighter>
          <ul style={{ marginTop: 10 }}>
            <li><b>Linealidad:</b> Relación lineal entre variables.</li>
            <li><b>Independencia:</b> Observaciones independientes.</li>
            <li><b>Homocedasticidad:</b> Varianza de errores constante.</li>
            <li><b>Normalidad:</b> Errores con distribución normal.</li>
          </ul>
          <div style={subtitle}><span>💡</span>Casos de Uso Detallados</div>
          <ul>
            <li><b>Finanzas:</b> Predecir el precio de una acción.</li>
            <li><b>Recursos Humanos:</b> Estimar el salario de un empleado.</li>
            <li><b>Marketing:</b> Decidir cómo distribuir el presupuesto publicitario.</li>
            <li><b>Ciencias Ambientales:</b> Predecir niveles de contaminación del aire.</li>
          </ul>
          <div style={subtitle}><span>📊</span>Tabla de Interpretación General</div>
          <div style={{ overflowX: 'auto', margin: '18px 0' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', border: '1px solid #e5e7eb' }}>
              <thead>
                <tr style={{ background: '#f9fafb' }}>
                  <th style={{ padding: '12px', border: '1px solid #e5e7eb', textAlign: 'left' }}>Métrica</th>
                  <th style={{ padding: '12px', border: '1px solid #e5e7eb', textAlign: 'left' }}>Rango / Valor</th>
                  <th style={{ padding: '12px', border: '1px solid #e5e7eb', textAlign: 'left' }}>Interpretación General</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>Coeficiente de Determinación (R²)</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>&gt; 0.80</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>💪 Muy Fuerte: El modelo explica una gran parte de la variabilidad.</td>
                </tr>
                <tr>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}></td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>0.60 - 0.79</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>👍 Fuerte: El modelo tiene un buen poder explicativo.</td>
                </tr>
                <tr>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}></td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>0.40 - 0.59</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>🙂 Moderado: El modelo es útil, pero podría haber otros factores importantes.</td>
                </tr>
                <tr>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}></td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>0.20 - 0.39</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>🤔 Débil: La relación es débil; el modelo no explica mucho.</td>
                </tr>
                <tr>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}></td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>&lt; 0.20</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>👎 Insignificante: El modelo no es mucho mejor que usar el promedio.</td>
                </tr>
                <tr>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>Error Cuadrático Medio (MSE)</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>No hay rangos universales</td>
                  <td style={{ padding: '12px', border: '1px solid #e5e7eb' }}>📉 Relativo: Su "bondad" depende de la escala de la variable que predices.</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div style={subtitle}><span>📝</span>Explicación Detallada</div>
          <div style={{ marginBottom: 16 }}>
            <b>1. Coeficiente de Determinación (R²)</b><br/>
            El R² es fácil de interpretar porque es un valor relativo (un porcentaje). Nos dice qué porción de la variabilidad de tu variable de resultado es explicada por el modelo. La tabla de arriba es una guía comúnmente aceptada.<br/><br/>
            <b>Ejemplo:</b> Un R² de 0.85 como en tu caso, cae en la categoría de "Muy Fuerte". Significa que el 85% de las diferencias observadas en tus datos de resultado se deben a la variable que usaste para predecir. Eso es, en casi cualquier campo, una excelente noticia.<br/><br/>
            <b>Advertencia:</b> Lo que se considera un "buen" R² puede variar según el campo de estudio. En física o química, se esperan valores muy altos (&gt;0.95). En ciencias sociales (como economía o psicología), un R² de 0.30 (30%) ya podría considerarse útil y significativo.
          </div>
          <div>
            <b>2. Error Cuadrático Medio (MSE)</b><br/>
            Aquí es donde el contexto es fundamental. El MSE es una medida absoluta del error y sus unidades son las unidades de tu variable de resultado, pero al cuadrado. No existen rangos universales de "bueno" o "malo" para el MSE.<br/><br/>
            Un MSE puede ser bueno o malo dependiendo de la escala de lo que estás midiendo.<br/><br/>
            <b>Ejemplo 1 (Malo):</b> Imagina que creas un modelo para predecir las calificaciones de estudiantes (en una escala de 0 a 5). Si tu modelo tiene un MSE de 4.0, es terrible. La raíz cuadrada (RMSE) sería 2.0, lo que significa que tus predicciones se equivocan, en promedio, por 2 puntos. En una escala de 5, eso es un error enorme.<br/><br/>
            <b>Ejemplo 2 (Bueno):</b> Ahora imagina que creas un modelo para predecir el precio de una casa en Ibagué (en pesos colombianos). Si tu modelo tiene un MSE de 4.000.000, puede sonar altísimo. Pero la raíz cuadrada (RMSE) es de 2.000.000 COP. Si el precio promedio de las casas es de 250.000.000 COP, un error promedio de 2 millones puede ser considerado muy aceptable.
          </div>
          <div style={subtitle}><span>🧑‍💻</span>Ejemplo de Código Comentado (Python)</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>
{codeLinear}
          </SyntaxHighlighter>
        </div>
      </div>
      <hr style={divider} />
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="target">🎯</span> <span style={highlight}>Regresión Logística: Clasificando el Mundo en Categorías</span>
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          A pesar de su nombre, la regresión logística es el algoritmo fundamental para la clasificación binaria. Responde a preguntas de tipo "Sí/No", "¿Es A o B?", "Verdadero o Falso".
          <div style={subtitle}><span>📚</span>Teoría Ampliada</div>
          La clave es la función sigmoide (o logística):
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 16, margin: '18px 0' }}>
{`σ(z) = 1 / (1 + e^(-z))`}
          </SyntaxHighlighter>
          Este valor se interpreta como la probabilidad de que la observación pertenezca a la clase positiva (clase "1"). Para clasificar, se establece un umbral de decisión (normalmente 0.5). La función de coste aquí es la <b style={highlight}>Entropía Cruzada Binaria (Log Loss)</b>.
          <div style={subtitle}><span>💡</span>Casos de Uso Detallados</div>
          <ul>
            <li><b>Medicina:</b> Diagnóstico de tumores.</li>
            <li><b>Detección de Fraude:</b> Transacciones con tarjeta de crédito.</li>
            <li><b>Marketing Digital:</b> Predicción de abandono de clientes.</li>
            <li><b>Riesgo crediticio:</b> Automatización de aprobación de préstamos.</li>
          </ul>
          <div style={subtitle}><span>🧑‍💻</span>Ejemplo de Código Comentado (Python)</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>
{codeLogistic}
          </SyntaxHighlighter>
        </div>
      </div>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="rocket">🚀</span> Implementación Práctica de Modelos con Scikit-learn
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <p>
            Scikit-learn es la biblioteca de facto para el machine learning en Python. Su éxito se debe a su API consistente, su amplia gama de algoritmos y sus herramientas integradas que simplifican todo el proceso de creación de un modelo.<br/><br/>
            El objeto central de Scikit-learn es el <b>Estimador</b>. Un estimador es cualquier objeto que aprende de los datos. Todos los estimadores comparten métodos consistentes:
          </p>
          <ul>
            <li><b>.fit(X, y):</b> El método de entrenamiento. Aprende los patrones a partir de las características X y la etiqueta y.</li>
            <li><b>.predict(X_new):</b> Una vez entrenado, genera predicciones para datos nuevos X_new.</li>
            <li><b>.transform(X)</b> o <b>.fit_transform(X):</b> Usado por los preprocesadores para limpiar, escalar o modificar los datos.</li>
          </ul>
          <p>El Flujo de Trabajo Estandarizado en 6 Pasos:</p>
          <ol>
            <li><b>Paso 1: Carga de Datos</b></li>
          </ol>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica1}</SyntaxHighlighter>
          <ol start={2}>
            <li><b>Paso 2: Preparación y División de los Datos</b></li>
          </ol>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica2}</SyntaxHighlighter>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica3}</SyntaxHighlighter>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica4}</SyntaxHighlighter>
          <ol start={3}>
            <li><b>Paso 3: Elegir y Crear una Instancia del Modelo</b></li>
          </ol>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica5}</SyntaxHighlighter>
          <ol start={4}>
            <li><b>Paso 4: Entrenar el Modelo</b></li>
          </ol>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica6}</SyntaxHighlighter>
          <ol start={5}>
            <li><b>Paso 5: Realizar Predicciones</b></li>
          </ol>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica7}</SyntaxHighlighter>
          <ol start={6}>
            <li><b>Paso 6: Evaluar el Modelo</b></li>
          </ol>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePractica8}</SyntaxHighlighter>
          <div style={{ margin: '24px 0 8px 0', fontWeight: 600, color: '#a259f7', fontSize: 18 }}>Resumen del Flujo de Trabajo en un Script Completo</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePracticaResumen}</SyntaxHighlighter>
        </div>
      </div>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="cherry blossom">🌸</span> Actividad Práctica: Tu Primer Clasificador de Flores
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Contexto del Problema:</b><br/>
          Eres un botánico de datos y has sido encargado de crear una herramienta que pueda clasificar automáticamente las especies de flores Iris basándose en las medidas de sus pétalos y sépalos. Utilizarás el famoso dataset "Iris".<br/><br/>
          <b>Objetivo:</b> Construir y evaluar un modelo de machine learning utilizando Regresión Logística y el flujo de trabajo de 6 pasos de Scikit-learn para predecir la especie de una flor.<br/><br/>
          <b>Instrucciones Paso a Paso:</b>
          <ol style={{ margin: '16px 0 0 20px' }}>
            <li><b>Paso 1:</b> Cargar y explorar los datos con <code>load_iris</code> de sklearn.datasets.</li>
            <li><b>Paso 2:</b> Preparar los datos, separar X e y, y dividir en entrenamiento/prueba.</li>
            <li><b>Paso 3:</b> Elegir y crear el modelo <code>LogisticRegression</code>.</li>
            <li><b>Paso 4:</b> Entrenar el modelo con <code>.fit()</code>.</li>
            <li><b>Paso 5:</b> Realizar predicciones con <code>.predict()</code>.</li>
            <li><b>Paso 6:</b> Evaluar el modelo con <code>accuracy_score</code> y <code>classification_report</code>.</li>
          </ol>
          <div style={{ margin: '18px 0' }}>
            <b>Preguntas para Reflexionar:</b>
            <ul>
              <li>¿Cuál fue la exactitud (accuracy) general de tu modelo? ¿Te parece un buen resultado?</li>
              <li>¿Hay alguna especie de flor que el modelo prediga mejor que las otras?</li>
              <li>¿Qué tan confiado estarías en la predicción para una nueva flor?</li>
            </ul>
            <b>Desafío Adicional:</b> Prueba con <code>KNeighborsClassifier</code> y compara resultados.
          </div>
          <details style={{ margin: '18px 0' }}>
            <summary style={{ cursor: 'pointer', color: '#a259f7', fontWeight: 600, fontSize: 17 }}>Haz clic aquí para ver la solución propuesta</summary>
            <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeIris}</SyntaxHighlighter>
          </details>
        </div>
      </div>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="gear">🛠️</span> Del Modelo al Servicio - Pipelines y APIs
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Introducción: ¿Por qué ir más allá de un script?</b><br/>
          En un entorno real, no basta con tener un modelo que funciona en un Jupyter Notebook. Necesitamos un sistema que sea:<br/>
          <ul>
            <li><b>Reproducible:</b> Cualquiera en tu equipo debe poder ejecutar tu código y obtener los mismos resultados.</li>
            <li><b>Robusto:</b> El sistema debe manejar los datos de manera consistente, desde el entrenamiento hasta la predicción.</li>
            <li><b>Desplegable:</b> El modelo final debe ser accesible para otros sistemas o usuarios a través de una interfaz, como una API.</li>
          </ul>
          Aquí es donde entran en juego los <b>Pipelines</b> y los <b>Servicios API</b>.<br/><br/>
          <b>1. Diseño de un Pipeline de Entrenamiento y Evaluación</b><br/>
          Un pipeline en Scikit-learn es un objeto que encapsula una secuencia de pasos de transformación y modelado. En lugar de aplicar cada paso manualmente (escalar, luego entrenar), los encadenas en un único objeto.<br/>
          <ul>
            <li><b>Simplicidad y Organización:</b> Condensa todo tu flujo de trabajo de preprocesamiento y modelado en un solo objeto.</li>
            <li><b>Prevención de Fuga de Datos (Data Leakage):</b> El pipeline asegura que los pasos de preprocesamiento se "ajusten" solo con los datos de entrenamiento y luego se apliquen a los de prueba.</li>
            <li><b>Facilidad de Uso:</b> Una vez construido, el pipeline se comporta como un modelo normal. Puedes usar <code>.fit()</code>, <code>.predict()</code> y <code>.score()</code> directamente sobre él.</li>
          </ul>
          <div style={subtitle}><span>🧑‍💻</span>Ejemplo Práctico: Construyendo un Pipeline</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codePipeline}</SyntaxHighlighter>
          <b>2. Creación de un Servicio API para Predicciones</b><br/>
          Un API permite que diferentes aplicaciones se comuniquen entre sí. Crearemos un servicio web que exponga un endpoint, reciba datos de una flor, cargue el pipeline y devuelva la predicción.<br/>
          <div style={subtitle}><span>🧑‍💻</span>Ejemplo de API con FastAPI</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeFastAPI}</SyntaxHighlighter>
          <div style={{ margin: '18px 0' }}>
            <b>¿Cómo Ejecutar y Probar tu API?</b><br/>
            <ol style={{ margin: '10px 0 0 20px' }}>
              <li>Asegúrate de tener los archivos <code>main.py</code> y <code>clasificador_iris_pipeline.joblib</code> en la misma carpeta.</li>
              <li>Ejecuta el servidor con: <code>uvicorn main:app --reload</code></li>
              <li>Abre <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer">http://127.0.0.1:8000/docs</a> para probar tu API con Swagger UI.</li>
            </ol>
            ¡Has desplegado exitosamente tu modelo de machine learning como un servicio profesional!
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

export default FundamentosML; 