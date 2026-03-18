import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =====================================================
# CARGAR DATOS
# =====================================================

ruta = r"C:\Users\heber\Desktop\proyecto 1\Cinetica de liberacion.xlsx"

df = pd.read_excel(ruta, header=None)

df.columns = df.iloc[0]
df = df[1:]

df["Tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
df["%liberacion"] = pd.to_numeric(df["%liberacion"], errors="coerce")

datos = df.dropna(subset=["Tiempo", "%liberacion"])
datos = datos[datos["Tiempo"] > 0]

t = datos["Tiempo"].values.astype(float)
M = datos["%liberacion"].values.astype(float)

n_data = len(M)

# =====================================================
# FUNCIONES ESTADÍSTICAS
# =====================================================

def rmse(obs, pred):
    return np.sqrt(np.mean((obs - pred)**2))

def aic(obs, pred, k):
    rss = np.sum((obs - pred)**2)
    return n_data * np.log(rss / n_data) + 2 * k

def bic(obs, pred, k):
    rss = np.sum((obs - pred)**2)
    return n_data * np.log(rss / n_data) + k * np.log(n_data)

# =====================================================
# MODELOS
# =====================================================

resultados = {}

# --- KP ---
log_t = np.log10(t)
log_M = np.log10(M)

slope, intercept, _, _, _ = linregress(log_t, log_M)
Kp = 10**intercept
n_kp = slope

M_kp = Kp * t**n_kp

resultados["Korsmeyer–Peppas"] = M_kp

# --- Higuchi ---
sqrt_t = np.sqrt(t)
slope, intercept, _, _, _ = linregress(sqrt_t, M)

Kh = slope
M_h = Kh * np.sqrt(t) + intercept

resultados["Higuchi"] = M_h

# --- Primer orden ---
M_remaining = 100 - M
mask = M_remaining > 0

t_po = t[mask]
ln_M = np.log(M_remaining[mask])

slope, intercept, _, _, _ = linregress(t_po, ln_M)

K1 = -slope
M_po = 100 * (1 - np.exp(-K1 * t))

resultados["Primer orden"] = M_po

# --- Weibull ---
F = M / 100
mask = (F > 0) & (F < 1)

t_w = t[mask]
F_w = F[mask]

X = np.log(t_w)
Y = np.log(-np.log(1 - F_w))

slope, intercept, _, _, _ = linregress(X, Y)

beta = slope
alpha = np.exp(-intercept / slope)

M_w = 100 * (1 - np.exp(-(t / alpha) ** beta))

resultados["Weibull"] = M_w

# =====================================================
# EVALUACIÓN ESTADÍSTICA
# =====================================================

print("\n🏆 COMPARACIÓN DE MODELOS\n")

tabla = []

for nombre, pred in resultados.items():

    r = rmse(M, pred)
    AIC = aic(M, pred, 2)
    BIC = bic(M, pred, 2)

    tabla.append([nombre, r, AIC, BIC])

    print(f"{nombre}")
    print(f"RMSE = {r:.3f}")
    print(f"AIC  = {AIC:.3f}")
    print(f"BIC  = {BIC:.3f}\n")

# Mejor modelo por AIC
mejor = min(tabla, key=lambda x: x[2])[0]

print("🥇 MEJOR MODELO (AIC):", mejor)

# =====================================================
# GRAFICA
# =====================================================

plt.figure(figsize=(9,6))

plt.scatter(t, M, color="black", s=70,
            label="Datos experimentales")

for nombre, pred in resultados.items():
    plt.plot(t, pred, label=nombre)

plt.xlabel("Tiempo (h)")
plt.ylabel("% liberación")
plt.title("Comparación de modelos — SNEDDS CPT")

plt.legend()
plt.grid(True)
plt.show()

# EJEMPLO DE PARÁMETROS (CAMBIA POR LOS TUYOS)

bioaccesibilidad = 0.75   # fracción
tamano_nm = 40            # nm
zeta = -12                # mV

# Factor de absorción dependiente de tamaño
f_tamano = np.exp(-tamano_nm / 200)

# Factor de estabilidad coloidal
f_zeta = 1 / (1 + np.exp(-(abs(zeta) - 10)))

# Predicción de absorción relativa
absorcion = bioaccesibilidad * f_tamano * f_zeta

print("\n💊 Absorción relativa estimada =", absorcion)