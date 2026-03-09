import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# 1. CARGAR EXCEL
# =========================

df = pd.read_excel(
    "Datos generales digestion.xlsx",
    sheet_name="Hoja1"
)

# Limpiar nombres de columnas
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

print("Columnas detectadas:")
print(df.columns)

# =========================
# 2. RENOMBRAR COLUMNAS CLAVE
# =========================

df = df.rename(columns={
    "Fase": "fase",
    "Tiempo_min": "tiempo_min",
    "Tamaño_de_particula": "tamano_nm"
})

# Verificar que existan
columnas_necesarias = ["fase", "tiempo_min", "tamano_nm"]

for col in columnas_necesarias:
    if col not in df.columns:
        raise ValueError(f"❌ Falta la columna obligatoria: {col}")

# =========================
# 3. LIMPIAR FASES
# =========================

df["fase"] = df["fase"].astype(str).str.strip()

# Corregir posible error tipográfico
df["fase"] = df["fase"].replace({
    "Incial": "Inicial"
})

print("\nFases detectadas:")
print(df["fase"].unique())

# =========================
# 4. FILTRAR INICIAL Y BOCA
# =========================

df_IB = df[df["fase"].isin(["Inicial", "Boca"])]

# =========================
# 5. AGRUPAR POR TIEMPO
# =========================

stats = (
    df_IB
    .groupby("tiempo_min")["tamano_nm"]
    .agg(["mean", "std"])
    .reset_index()
)

print("\nEstadísticos:")
print(stats)

# =========================
# 6. MODELO DE ADSORCIÓN
# =========================

def modelo_adsorcion(t, Dmax, k):
    D0 = stats["mean"].iloc[0]
    return Dmax - (Dmax - D0) * np.exp(-k * t)

t_data = stats["tiempo_min"].values
D_data = stats["mean"].values

# Ajuste del modelo (YA CORREGIDO)
params, _ = curve_fit(modelo_adsorcion, t_data, D_data)

Dmax_fit, k_fit = params

print("\nParámetros estimados:")
print("Dmax =", Dmax_fit)
print("k_ads =", k_fit)

# =========================
# 7. GRAFICA
# =========================

t_fit = np.linspace(min(t_data), max(t_data), 100)
D_fit = modelo_adsorcion(t_fit, Dmax_fit, k_fit)

plt.errorbar(
    stats["tiempo_min"],
    stats["mean"],
    yerr=stats["std"],
    fmt='o',
    capsize=5,
    label="Datos experimentales"
)

plt.plot(t_fit, D_fit, label="Modelo exponencial")

plt.xlabel("Tiempo (min)")
plt.ylabel("Tamaño de partícula (nm)")
plt.title("Fase Inicial → Boca")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# =========================
# 1. CARGAR EXCEL
# =========================

df = pd.read_excel("Datos generales digestion.xlsx", sheet_name="Hoja1")

print("Columnas originales:")
print(df.columns.tolist())

# =========================
# 2. NORMALIZAR COLUMNAS
# =========================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("°", "")
    .str.replace("(", "")
    .str.replace(")", "")
    .str.replace("ñ", "n")
    .str.replace("á", "a")
    .str.replace("é", "e")
    .str.replace("í", "i")
    .str.replace("ó", "o")
    .str.replace("ú", "u")
)

print("\nColumnas normalizadas:")
print(df.columns.tolist())

# =========================
# 3. DETECTAR COLUMNAS
# =========================

col_fase = [c for c in df.columns if "fase" in c.lower()][0]
col_tiempo = [c for c in df.columns if "tiempo" in c.lower()][0]
col_tamano = [c for c in df.columns if "particula" in c.lower()][0]

print("\nColumnas detectadas:")
print("Fase:", col_fase)
print("Tiempo:", col_tiempo)
print("Tamaño:", col_tamano)

# =========================
# 4. FILTRAR ESTOMAGO + TRANSICION
# =========================

df_est = df[
    df[col_fase]
    .astype(str)
    .str.contains("estomago", case=False, na=False)
].copy()

if df_est.empty:
    raise ValueError("❌ No se encontraron datos de Estómago.")

# convertir tiempo a numero
df_est[col_tiempo] = pd.to_numeric(df_est[col_tiempo], errors="coerce")

# eliminar NaN
df_est = df_est.dropna(subset=[col_tiempo, col_tamano])

# =========================
# 5. PROMEDIAR REPLICADOS
# =========================

df_est = (
    df_est
    .groupby(col_tiempo)[col_tamano]
    .mean()
    .reset_index()
)

df_est = df_est.sort_values(col_tiempo)

t_est = df_est[col_tiempo].values.astype(float)
D_est = df_est[col_tamano].values.astype(float)

print("\nTiempos estomago:", t_est)
print("Tamaño promedio:", D_est)

# =========================
# 6. MODELO EXPONENCIAL
# =========================

def modelo_estomago(t, D_eq, k):
    D0 = D_est[0]
    return D_eq + (D0 - D_eq) * np.exp(-k * t)

p0 = [D_est[-1], 0.001]

params_est, _ = curve_fit(
    modelo_estomago,
    t_est,
    D_est,
    p0=p0,
    maxfev=10000
)

D_eq_est, k_est = params_est

print("\nRESULTADOS ESTOMAGO")
print("D_eq =", D_eq_est)
print("k =", k_est)

# =========================
# 7. R²
# =========================

D_pred = modelo_estomago(t_est, D_eq_est, k_est)

ss_res = np.sum((D_est - D_pred) ** 2)
ss_tot = np.sum((D_est - np.mean(D_est)) ** 2)

r2 = 1 - (ss_res / ss_tot)

print("R² =", r2)

# =========================
# 8. GRAFICA
# =========================

t_fit = np.linspace(min(t_est), max(t_est), 200)
D_fit = modelo_estomago(t_fit, D_eq_est, k_est)

plt.figure()

plt.scatter(t_est, D_est, label="Datos experimentales")

plt.plot(t_est, D_est, linestyle="--", alpha=0.6)

plt.plot(t_fit, D_fit, linewidth=2, label="Modelo exponencial")

plt.xlabel("Tiempo (min)")
plt.ylabel("Tamaño de partícula (nm)")
plt.title("Cinética en fase Estómago (incluye transición)")
plt.legend()

plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. CARGAR EXCEL
# =========================

df = pd.read_excel("Datos generales digestion.xlsx")

# =========================
# 2. LIMPIAR NOMBRES
# =========================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

print("Columnas detectadas:")
print(df.columns)

# =========================
# 3. DEFINIR COLUMNAS
# =========================

col_fase = "Fase"
col_tamano = "Tamaño_de_particula"
col_tiempo = "Tiempo_min"

# =========================
# 4. FILTRAR SOLO INTESTINO
# =========================

df_int = df[
    df[col_fase].str.contains("Intestino", case=False, na=False)
].copy()

# convertir a número
df_int[col_tamano] = pd.to_numeric(df_int[col_tamano], errors="coerce")
df_int[col_tiempo] = pd.to_numeric(df_int[col_tiempo], errors="coerce")

# eliminar vacíos
df_int = df_int.dropna(subset=[col_tamano, col_tiempo])

# =========================
# 5. CALCULAR PROMEDIO Y DESVIACIÓN
# =========================

datos = (
    df_int
    .groupby(col_tiempo)[col_tamano]
    .agg(['mean','std'])
    .reset_index()
)

datos = datos.sort_values(col_tiempo)

print("\nPromedios y desviación estándar:")
print(datos)

# =========================
# 6. VARIABLES PARA GRAFICA
# =========================

t = datos[col_tiempo]
tam_prom = datos['mean']
tam_std = datos['std']

# =========================
# 7. GRAFICA CON BARRAS DE ERROR
# =========================

plt.figure()

plt.errorbar(
    t,
    tam_prom,
    yerr=tam_std,
    fmt='o-',
    capsize=5,
    label="Promedio ± desviación estándar"
)

plt.xlabel("Tiempo de digestión (min)")
plt.ylabel("Tamaño de partícula (nm)")
plt.title("Estabilidad del SNEDDS en fase intestinal")

plt.grid(True)
plt.legend()

plt.show()