# Importar librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

# Cargar datos experimentales
df = pd.read_excel('digestion.xlsx')

# Mostrar nombres exactos de las columnas para depuración
print('Columnas disponibles en el archivo:', df.columns.tolist())

# Codificar variables categóricas
le_fase = LabelEncoder()
le_enzima = LabelEncoder()
df['fase'] = le_fase.fit_transform(df['fase'])
df['enzima'] = le_enzima.fit_transform(df['enzima'])

# Visualizar transformaciones estructurales

# Gráfica 1: Tamaño de partícula vs tiempo por fase
plt.figure(figsize=(10,6))
fase_labels = {0: 'inicial', 1: 'boca', 2: 'estómago', 3: 'intestino'}
orden_fases = [0, 1, 2, 3]
for fase in orden_fases:
	if fase in df['fase'].unique():
		datos_fase = df[df['fase'] == fase]
		datos_fase = datos_fase.sort_values('tiempo')
		plt.plot(datos_fase['tiempo'], datos_fase['tamaño_particula'], marker='o', linestyle='-', label=fase_labels[fase])
plt.xlabel('Tiempo (min)')
plt.ylabel('Tamaño de partícula (nm)')
plt.title('Tamaño de partícula de SNEDDS durante digestión')
plt.legend(title='Fase', loc='best')
plt.show()
plt.xlabel('Tiempo (min)')
plt.ylabel('Tamaño de partícula (nm)')
plt.title('Tamaño de partícula de SNEDDS durante digestión')
plt.legend()
plt.show()

# Gráfica 2: Índice de polidispersidad vs tiempo por fase
plt.figure(figsize=(10,6))
for fase in orden_fases:
	if fase in df['fase'].unique():
		datos_fase = df[df['fase'] == fase]
		datos_fase = datos_fase.sort_values('tiempo')
		plt.plot(datos_fase['tiempo'], datos_fase['indice_polidispersidad'], marker='o', linestyle='-', label=fase_labels[fase])
plt.xlabel('Tiempo (min)')
plt.ylabel('Índice de polidispersidad (PDI)')
plt.title('Índice de polidispersidad de SNEDDS durante digestión')
plt.legend(title='Fase', loc='best')
plt.show()
plt.xlabel('Tiempo (min)')
plt.ylabel('Índice de polidispersidad (PDI)')
plt.title('Índice de polidispersidad de SNEDDS durante digestión')
plt.legend()
plt.show()

# Seleccionar variables de entrada y salida
X = df[['fase', 'tiempo', 'tamaño_particula', 'indice_polidispersidad', 'enzima', 'ph', 'agitacion', 'temperatura']]
y = df['tamaño_particula']  # Cambia aquí si quieres predecir otra columna existente

# Escalar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir modelo de red neuronal
model = keras.Sequential([
	keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
	keras.layers.Dense(16, activation='relu'),
	keras.layers.Dense(1)  # Salida para regresión
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar modelo
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Evaluar modelo
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")

# Hacer predicciones
predicciones = model.predict(X_test)
print(predicciones)

# Visualizar predicción vs. real
plt.figure(figsize=(8,5))
plt.scatter(y_test, predicciones, alpha=0.7)
plt.xlabel('Valor real')
plt.ylabel('Predicción')
plt.title('Predicción de capacidad funcional SNEDDS')
plt.show()
# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

# Cargar datos desde Excel
df = pd.read_excel('digestion.xlsx')

# Mostrar primeras filas para verificar
print(df.head())

# Codificar variables categóricas (fase, enzimas)
le_fase = LabelEncoder()
le_enzima = LabelEncoder()
df['fase'] = le_fase.fit_transform(df['fase'])
df['enzima'] = le_enzima.fit_transform(df['enzima'])

# Seleccionar variables de entrada y salida
X = df[['fase', 'tiempo', 'tamaño_particula', 'indice_polidispersidad', 'enzima', 'pH']]
y = df['bioaccesibilidad']  # Cambia si tu columna objetivo tiene otro nombre

# Escalar variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir modelo de red neuronal
model = keras.Sequential([
	keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
	keras.layers.Dense(16, activation='relu'),
	keras.layers.Dense(1)  # Salida para regresión
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar modelo
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

# Evaluar modelo
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")

# Hacer predicciones
predicciones = model.predict(X_test)
print(predicciones)
