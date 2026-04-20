import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# CARGAR DATOS
# =========================
df = pd.read_csv("digestion.csv")

# Limpiar columnas
df.columns = df.columns.str.lower().str.strip()

print("Columnas detectadas:", df.columns)

# =========================
# RENOMBRAR VARIABLES
# =========================
df = df.rename(columns={
    "tiempo": "tiempo_min",
    "tamaño_particula": "tamano_nm"
})

# =========================
# NORMALIZAR FASES
# =========================
df["fase"] = df["fase"].str.lower().str.strip()

# =========================
# AJUSTE DE TIEMPO (CLAVE 🔥)
# =========================
def ajustar_tiempo(row):
    if "boca" in row["fase"]:
        return 2
    elif "estomago" in row["fase"] or "estómago" in row["fase"]:
        return 30 + row["tiempo_min"]   # empieza en 30 min
    elif "intestino" in row["fase"]:
        return 120 + row["tiempo_min"]  # empieza en 120 min
    else:
        return 0

df["tiempo_ajustado"] = df.apply(ajustar_tiempo, axis=1)

# =========================
# FILTRAR FASES
# =========================
estomago = df[df["fase"].str.contains("estomago|estómago")]
intestino = df[df["fase"].str.contains("intestino")]

# =========================
# MODELOS
# =========================
def modelo_estomago(t, Deq, k):
    D0 = estomago["tamano_nm"].iloc[0]
    return Deq + (D0 - Deq) * np.exp(-k * t)

def modelo_intestino(t, a, b):
    return a * t + b

# =========================
# AJUSTE
# =========================
popt_estomago, _ = curve_fit(
    modelo_estomago,
    estomago["tiempo_ajustado"],
    estomago["tamano_nm"],
    maxfev=10000
)

popt_intestino, _ = curve_fit(
    modelo_intestino,
    intestino["tiempo_ajustado"],
    intestino["tamano_nm"]
)

# =========================
# GRÁFICA PROFESIONAL
# =========================
plt.figure(figsize=(10,6))

# Datos experimentales
plt.scatter(
    df["tiempo_ajustado"],
    df["tamano_nm"],
    label="Datos experimentales"
)

# Curva estómago
t_e = np.linspace(
    estomago["tiempo_ajustado"].min(),
    estomago["tiempo_ajustado"].max(),
    100
)

plt.plot(
    t_e,
    modelo_estomago(t_e, *popt_estomago),
    label="Modelo Estómago"
)

# Curva intestino
t_i = np.linspace(
    intestino["tiempo_ajustado"].min(),
    intestino["tiempo_ajustado"].max(),
    100
)

plt.plot(
    t_i,
    modelo_intestino(t_i, *popt_intestino),
    label="Modelo Intestino"
)

# =========================
# FORMATO FINAL
# =========================
plt.title("Modelado del tamaño de partícula durante la digestión in vitro")
plt.xlabel("Tiempo (min)")
plt.ylabel("Tamaño de partícula (nm)")

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()