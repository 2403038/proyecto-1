import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.integrate import odeint
from scipy.interpolate import interp1d

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
# MODELOS DE LIBERACIÓN
# =====================================================
resultados = {}

# --- Korsmeyer–Peppas ---
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
    AIC_val = aic(M, pred, 2)
    BIC_val = bic(M, pred, 2)
    tabla.append([nombre, r, AIC_val, BIC_val])
    print(f"{nombre}")
    print(f"RMSE = {r:.3f}")
    print(f"AIC  = {AIC_val:.3f}")
    print(f"BIC  = {BIC_val:.3f}\n")

mejor = min(tabla, key=lambda x: x[2])[0]
print("🥇 MEJOR MODELO (AIC):", mejor)

# =====================================================
# GRAFICA COMPARATIVA
# =====================================================
plt.figure(figsize=(9,6))
plt.scatter(t, M, color="black", s=70, label="Datos experimentales")
for nombre, pred in resultados.items():
    plt.plot(t, pred, label=nombre)
plt.xlabel("Tiempo (h)")
plt.ylabel("% liberación")
plt.title("Comparación de modelos — SNEDDS CPT")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# PARÁMETROS DE FORMULACIÓN
# =====================================================
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

# =====================================================
# SIMULACIÓN MECANÍSTICA DE DIGESTIÓN
# =====================================================
# Datos de lipólisis (% liberación de ácidos grasos)
tiempo_lip = np.array([0, 15, 30, 45, 60, 90, 120])
lip_data = np.array([0, 10, 25, 40, 55, 70, 85])
lip_func = interp1d(tiempo_lip, lip_data/100, kind='linear', fill_value="extrapolate")

# Parámetros del modelo
k_g = 0.05    # vaciamiento gástrico
k_abs = 0.12  # absorción intestinal (nanoemulsión 40nm + PEG-folato)
k_elim = 0.02 # aclaramiento plasmático

C0_st = 1.0
C0_int = 0.0
C0_plasma = 0.0

# Modelo diferencial
def modelo(C, t):
    C_st, C_int, C_plasma = C
    k_lip = lip_func(t)
    dC_st_dt = -k_g * C_st
    dC_int_dt = k_g * C_st - k_abs * C_int - k_lip * C_int
    dC_plasma_dt = k_abs * C_int - k_elim * C_plasma
    return [dC_st_dt, dC_int_dt, dC_plasma_dt]

# Simulación
t_sim = np.linspace(0, 120, 500)
C = odeint(modelo, [C0_st, C0_int, C0_plasma], t_sim)
C_st, C_int, C_plasma = C.T

# Gráfico
plt.figure(figsize=(8,5))
plt.plot(t_sim, C_st, label='Estómago')
plt.plot(t_sim, C_int, label='Intestino')
plt.plot(t_sim, C_plasma, label='Plasma')
plt.xlabel('Tiempo (min)')
plt.ylabel('Concentración relativa')
plt.title('Simulación digestión-absorción de la nanoemulsión')
plt.legend()
plt.grid(True)
plt.show()

# Cálculo manual de biodisponibilidad relativa (tipo trapecio)
AUC = sum((C_plasma[i] + C_plasma[i+1])/2 * (t_sim[i+1] - t_sim[i]) for i in range(len(t_sim)-1))
print(f"Biodisponibilidad relativa (AUC simulada): {AUC:.3f}")

# =====================================================
# SIMULACIÓN: EFECTO DE LA LIPÓLISIS EN LA BIODISPONIBILIDAD
# =====================================================
lipolisis_range = np.linspace(0, 1, 50)  # Fracción de lipólisis de 0 a 100%

# Biodisponibilidad relativa simulada según lipólisis
F_sim = bioaccesibilidad * lipolisis_range * f_tamano * f_zeta

plt.figure(figsize=(8,5))
plt.plot(lipolisis_range*100, F_sim, color='darkorange', lw=2)
plt.xlabel("Fracción de lipólisis (%)")
plt.ylabel("Biodisponibilidad relativa")
plt.title("Impacto de la digestión lipídica en la absorción")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

# 1️⃣ Liberación experimental vs modelo Higuchi
plt.subplot(2,1,1)
plt.scatter(t, M, color='black', s=50, label='Datos experimentales')
plt.plot(t, M_h, color='blue', lw=2, label='Modelo Higuchi')
plt.xlabel("Tiempo (h)")
plt.ylabel("% liberación")
plt.title("Liberación de Camptotecina — SNEDDS")
plt.legend()
plt.grid(True)

# 2️⃣ Simulación digestión-absorción vs lipólisis
plt.subplot(2,1,2)
# Concentración plasmática simulada
plt.plot(t_sim, C_plasma, color='green', lw=2, label='Plasma (simulación ODE)')
# Lipólisis vs biodisponibilidad
plt.plot(lipolisis_range*120, F_sim*max(C_plasma), '--', color='orange', lw=2, label='Efecto lipólisis')
plt.xlabel("Tiempo / Lipólisis (%)")
plt.ylabel("Concentración relativa / AUC")
plt.title("Simulación digestión-absorción y efecto de lipólisis")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()