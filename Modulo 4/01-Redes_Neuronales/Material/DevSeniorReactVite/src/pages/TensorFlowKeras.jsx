import React from 'react';
import Slidebar from '../components/Slidebar';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const codeTensorFlow = `import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- 1. Cargar y Preprocesar los Datos ---

# Cargamos el dataset. Nos quedamos con las 10,000 palabras más frecuentes.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Las reseñas son listas de números (índices de palabras) y tienen longitudes diferentes.
# Las redes neuronales necesitan entradas de tamaño uniforme.
# Usamos "pad_sequences" para que todas las reseñas tengan una longitud de 256 palabras.
# Si una reseña es más corta, se rellena con ceros. Si es más larga, se trunca.
train_data = pad_sequences(train_data, value=0, padding='post', maxlen=256)
test_data = pad_sequences(test_data, value=0, padding='post', maxlen=256)

# --- 2. Construir la Red Neuronal ---

# Usamos el modelo "Sequential", que es una pila lineal de capas.
model = keras.Sequential([
    # Capa de Embedding: Esta capa es especial para texto. Convierte los índices de palabras
    # en vectores densos de un tamaño fijo (16 en este caso). Aprende asociaciones de palabras.
    keras.layers.Embedding(10000, 16),
    
    # Capa GlobalAveragePooling1D: Promedia los vectores de palabras para cada reseña.
    # Esto nos da una salida de tamaño fijo para cada ejemplo, independientemente de la longitud.
    keras.layers.GlobalAveragePooling1D(),
    
    # Capa Densa (Oculta): Nuestra primera capa oculta con 16 neuronas.
    # Usa la activación ReLU para introducir no-linealidad.
    keras.layers.Dense(16, activation='relu'),
    
    # Capa Densa (Salida): La capa final. Tiene 1 neurona porque es una clasificación binaria.
    # Usa la activación Sigmoide para dar una salida entre 0 y 1 (probabilidad).
    keras.layers.Dense(1, activation='sigmoid')
])

# Mostramos un resumen de la arquitectura de nuestro modelo.
model.summary()

# --- 3. Compilar el Modelo ---

# Antes de entrenar, debemos configurar el proceso de aprendizaje.
model.compile(optimizer='adam',              # Algoritmo para minimizar la pérdida (Adam es muy popular).
              loss='binary_crossentropy', # Función de pérdida para clasificación binaria.
              metrics=['accuracy'])         # Métrica para monitorear el rendimiento (exactitud).

# --- 4. Entrenar el Modelo ---

# Separamos una parte de los datos de entrenamiento para validación.
# La validación nos ayuda a ver qué tal generaliza el modelo con datos que no ha visto.
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Entrenamos el modelo.
# epochs: Número de veces que el modelo verá todo el conjunto de datos de entrenamiento.
# batch_size: Número de ejemplos que se usan en una sola pasada para actualizar los pesos.
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# --- 5. Evaluar el Modelo ---

# Una vez entrenado, evaluamos su rendimiento en el conjunto de prueba (datos nunca vistos).
results = model.evaluate(test_data, test_labels, verbose=2)
print("\\nPérdida en prueba:", results[0])
print("Exactitud en prueba:", results[1])

# --- 6. Hacer Predicciones ---

# Podemos usar el modelo entrenado para predecir en nuevos datos.
predictions = model.predict(test_data)
print("\\nPredicción para la primera reseña de prueba:", predictions[0])
print("Etiqueta real:", test_labels[0])
# Si la predicción es > 0.5, es positiva. Si es < 0.5, es negativa.`;

const codeSimpleNN = `# Red Neuronal Simple para Clasificación de Flores Iris
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. Cargar y Preparar los Datos ---
iris = load_iris()
X, y = iris.data, iris.target

# Solo usamos 2 clases para simplificar (clasificación binaria)
X = X[y != 2]
y = y[y != 2]

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. Construir la Red Neuronal ---
model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(4,)),  # Capa oculta
    keras.layers.Dense(1, activation='sigmoid')                  # Capa de salida
])

# --- 3. Compilar y Entrenar ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

# --- 4. Evaluar ---
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Exactitud en prueba: {test_accuracy:.3f}")`;

const codeVisualization = `import matplotlib.pyplot as plt
import numpy as np

# Visualizar el entrenamiento
plt.figure(figsize=(12, 4))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de exactitud
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Exactitud de Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Exactitud de Validación')
plt.title('Exactitud durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()

plt.tight_layout()
plt.show()`;

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

const TensorFlowKeras = ({ isDarkTheme, onToggleTheme }) => (
  <div className="admin-panel">
    <Slidebar open={true} onClose={() => {}} isDarkTheme={isDarkTheme} alwaysVisible={true} />
    <div className="main-content" style={{ marginLeft: 260, maxWidth: '100%', background: 'transparent', minHeight: '100vh', paddingBottom: 60 }}>
      <div style={sectionCard}>
        <div style={accentTitle}>
          <span role="img" aria-label="robot">🤖</span> TensorFlow & Keras
        </div>
        <div style={{ fontSize: 17, lineHeight: 1.7 }}>
          <b>Introducción: Construcción de una Red Neuronal Simple</b><br/>
          Ahora, vamos a la práctica. Keras es una API de alto nivel dentro de TensorFlow que hace que construir 
          redes neuronales sea muy intuitivo. Vamos a construir una red neuronal que clasifique reseñas de películas 
          como "positivas" o "negativas" utilizando el famoso conjunto de datos IMDB.<br/><br/>

          <div style={subtitle}>📦 Instalación</div>
          Si no tienes TensorFlow instalado, abre una terminal o línea de comandos y ejecuta:<br/><br/>

          <div style={{ 
            background: '#2d3748', 
            color: '#e2e8f0', 
            padding: '15px', 
            borderRadius: '8px', 
            fontFamily: 'monospace',
            marginBottom: '20px'
          }}>
            pip install tensorflow
          </div>

          <div style={subtitle}>🎬 Problema: Clasificación de Reseñas de Películas</div>
          Vamos a construir una red neuronal que clasifique reseñas de películas como "positivas" o "negativas" 
          utilizando el dataset IMDB. Este es un problema de clasificación binaria perfecto para empezar.<br/><br/>

          <div style={subtitle}>🧑‍💻 Código Completo con Explicaciones</div>
          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeTensorFlow}</SyntaxHighlighter>

          <div style={subtitle}>🔍 Explicación Paso a Paso</div>
          
          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>1. Carga y Preprocesamiento de Datos</h4>
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              <li><b>Dataset IMDB:</b> 50,000 reseñas de películas (25,000 para entrenamiento, 25,000 para prueba)</li>
              <li><b>Vocabulario:</b> Limitado a las 10,000 palabras más frecuentes</li>
              <li><b>Padding:</b> Todas las reseñas se ajustan a 256 palabras para uniformidad</li>
            </ul>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>2. Arquitectura de la Red</h4>
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              <li><b>Embedding:</b> Convierte palabras en vectores densos de 16 dimensiones</li>
              <li><b>GlobalAveragePooling1D:</b> Promedia los vectores de palabras de cada reseña</li>
              <li><b>Dense(16, relu):</b> Capa oculta con 16 neuronas y activación ReLU</li>
              <li><b>Dense(1, sigmoid):</b> Capa de salida con 1 neurona y activación sigmoide</li>
            </ul>
          </div>

          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <h4 style={{ color: '#495057', marginBottom: '15px' }}>3. Compilación del Modelo</h4>
            <ul style={{ margin: 0, paddingLeft: '20px' }}>
              <li><b>Optimizer:</b> Adam (algoritmo de optimización adaptativo)</li>
              <li><b>Loss:</b> binary_crossentropy (función de pérdida para clasificación binaria)</li>
              <li><b>Metrics:</b> accuracy (métrica para monitorear el rendimiento)</li>
            </ul>
          </div>

          <div style={subtitle}>🎯 Ejemplo Más Simple: Clasificación de Flores</div>
          Para entender mejor los conceptos, aquí tienes un ejemplo más simple usando el dataset Iris:<br/><br/>

          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeSimpleNN}</SyntaxHighlighter>

          <div style={subtitle}>📊 Visualización del Entrenamiento</div>
          Es importante monitorear el entrenamiento para detectar overfitting:<br/><br/>

          <SyntaxHighlighter language="python" style={atomDark} customStyle={{ borderRadius: 10, fontSize: 14, margin: '18px 0' }}>{codeVisualization}</SyntaxHighlighter>

          <div style={subtitle}>🔧 Conceptos Clave de Keras</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#e8f5e8', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#2d5a2d', marginBottom: '10px' }}>🏗️ Modelo Sequential</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Pila lineal de capas</li>
                <li>Fácil de construir y entender</li>
                <li>Ideal para la mayoría de problemas</li>
                <li>No permite conexiones complejas</li>
              </ul>
            </div>
            <div style={{ background: '#fff3cd', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#856404', marginBottom: '10px' }}>⚙️ Compilación</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>Define el optimizador</li>
                <li>Especifica la función de pérdida</li>
                <li>Configura las métricas</li>
                <li>Prepara el modelo para entrenamiento</li>
              </ul>
            </div>
          </div>

          <div style={subtitle}>📈 Hiperparámetros Importantes</div>
          
          <div style={{ 
            background: '#f8f9fa', 
            padding: '20px', 
            borderRadius: '8px', 
            border: '1px solid #e9ecef',
            marginBottom: '20px'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div>
                <h4 style={{ color: '#495057', marginBottom: '10px' }}>🎯 Epochs</h4>
                <p style={{ margin: 0, lineHeight: '1.6' }}>
                  Número de veces que el modelo ve todo el dataset. Demasiados pueden causar overfitting.
                </p>
              </div>
              <div>
                <h4 style={{ color: '#495057', marginBottom: '10px' }}>📦 Batch Size</h4>
                <p style={{ margin: 0, lineHeight: '1.6' }}>
                  Número de ejemplos procesados antes de actualizar los pesos. Afecta la velocidad y estabilidad.
                </p>
              </div>
              <div>
                <h4 style={{ color: '#495057', marginBottom: '10px' }}>🔍 Validation Split</h4>
                <p style={{ margin: 0, lineHeight: '1.6' }}>
                  Porcentaje de datos de entrenamiento usado para validación. Ayuda a detectar overfitting.
                </p>
              </div>
              <div>
                <h4 style={{ color: '#495057', marginBottom: '10px' }}>⚡ Learning Rate</h4>
                <p style={{ margin: 0, lineHeight: '1.6' }}>
                  Tamaño de los pasos en la optimización. Demasiado alto puede causar inestabilidad.
                </p>
              </div>
            </div>
          </div>

          <div style={subtitle}>⚠️ Problemas Comunes y Soluciones</div>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
            <div style={{ background: '#ffe6e6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#dc2626', marginBottom: '10px' }}>❌ Overfitting</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>El modelo memoriza los datos de entrenamiento</li>
                <li>Buen rendimiento en entrenamiento, malo en prueba</li>
                <li><strong>Solución:</strong> Dropout, regularización, más datos</li>
              </ul>
            </div>
            <div style={{ background: '#ffe6e6', padding: '15px', borderRadius: '8px' }}>
              <h4 style={{ color: '#dc2626', marginBottom: '10px' }}>❌ Underfitting</h4>
              <ul style={{ margin: 0, paddingLeft: '20px' }}>
                <li>El modelo es demasiado simple</li>
                <li>Mal rendimiento en entrenamiento y prueba</li>
                <li><strong>Solución:</strong> Más capas, más neuronas, más épocas</li>
              </ul>
            </div>
          </div>

          <div style={{ 
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
            color: 'white', 
            padding: '20px', 
            borderRadius: '12px',
            marginTop: '30px'
          }}>
            <h3 style={{ marginBottom: '15px' }}>🚀 Próximos Pasos</h3>
            <ul style={{ margin: 0, paddingLeft: '20px', lineHeight: '1.6' }}>
              <li>Experimenta con diferentes arquitecturas de red</li>
              <li>Prueba diferentes optimizadores (SGD, RMSprop, Adam)</li>
              <li>Añade regularización (Dropout, L1/L2)</li>
              <li>Implementa early stopping para prevenir overfitting</li>
              <li>Explora técnicas de data augmentation</li>
              <li>Considera usar callbacks para monitoreo avanzado</li>
            </ul>
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

export default TensorFlowKeras; 