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
