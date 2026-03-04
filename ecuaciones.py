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