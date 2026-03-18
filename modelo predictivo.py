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





import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATOS DE EJEMPLO
# =========================
# Supongamos que tenemos un DataFrame con tus experimentos
# Cada fila = una formulación
data = pd.DataFrame({
    'lipido': [1,2,1,2,1,2],
    'surfactante':[1,1,2,2,1,2],
    'pH':[1.5, 1.5, 6.8, 6.8, 1.5, 6.8],
    'enzima':[0.8,0.8,1.0,1.0,0.8,1.0],
    'tamano':[40,40,60,60,40,60],
    'PEG_folato':[1,0,1,0,1,0],
    'Kp':[0.25,0.18,0.30,0.20,0.22,0.19],
    'n':[0.45,0.50,0.40,0.48,0.42,0.46],
    'liberacion':[60,50,70,55,65,52]
})

# =========================
# DIVIDIR DATOS
# =========================
X = data[['lipido','surfactante','pH','enzima','tamano','PEG_folato']]
y = data[['liberacion','n','Kp']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================
# ENTRENAR MODELO
# =========================
rf = RandomForestRegressor(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)

# =========================
# PREDICCIONES
# =========================
y_pred = rf.predict(X_test)

# =========================
# EVALUACIÓN
# =========================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R2 = {r2:.3f}, RMSE = {rmse:.3f}")

# =========================
# IMPORTANCIA DE VARIABLES
# =========================
importances = rf.feature_importances_
plt.figure(figsize=(8,5))
plt.bar(X.columns, importances, color='skyblue')
plt.ylabel("Importancia relativa")
plt.title("Jerarquía de variables en la liberación de SNEDDS")
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 📂 RUTAS DE ARCHIVOS
# ============================================================

ruta_liberacion = r"C:\Users\heber\Desktop\proyecto 1\Cinetica de liberacion.xlsx"
ruta_formulacion = r"C:\Users\heber\Desktop\proyecto 1\formulaciones.xlsx"

# ============================================================
# 🔵 PARTE 1 — MODELO DE LIBERACIÓN
# ============================================================

print("\n================ LIBERACIÓN =================")

df = pd.read_excel(ruta_liberacion, header=None)

# Usar primera fila como encabezado
df.columns = df.iloc[0]
df = df[1:]

# Convertir a numérico
df["Tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
df["%liberacion"] = pd.to_numeric(df["%liberacion"], errors="coerce")

# Variables adicionales (si no existen → valores constantes)
df["tamano"] = pd.to_numeric(df.get("tamano", 40), errors="coerce")
df["PEG_folato"] = pd.to_numeric(df.get("PEG_folato", 1), errors="coerce")
df["enzima"] = pd.to_numeric(df.get("enzima", 1.0), errors="coerce")
df["pH"] = pd.to_numeric(df.get("pH", 1.5), errors="coerce")

datos = df.dropna()

X1 = datos[['tamano','PEG_folato','enzima','pH']]
y1 = datos['%liberacion']

X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.3, random_state=42
)

rf1 = RandomForestRegressor(n_estimators=500, random_state=42)
rf1.fit(X_train, y_train)

y_pred1 = rf1.predict(X_test)

r2_1 = r2_score(y_test, y_pred1)
rmse_1 = np.sqrt(mean_squared_error(y_test, y_pred1))

print(f"R² liberación = {r2_1:.3f}")
print(f"RMSE liberación = {rmse_1:.3f}")

# Importancia variables liberación
plt.figure()
plt.bar(X1.columns, rf1.feature_importances_)
plt.title("Factores que controlan la liberación")
plt.ylabel("Importancia")
plt.show()

# Predicción vs real liberación
plt.figure()
plt.scatter(y_test, y_pred1)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Liberación experimental (%)")
plt.ylabel("Predicción (%)")
plt.title("Predicción vs Real — Liberación")
plt.show()

# ============================================================
# 🔴 PARTE 2 — MODELO DE FORMULACIÓN (DoE)
# ============================================================

print("\n================ FORMULACIÓN =================")

data = pd.read_excel(ruta_formulacion)

# Limpiar nombres de columnas
data.columns = (
    data.columns.astype(str)
    .str.strip()
    .str.replace('\n', ' ', regex=False)
    .str.replace(r'\s+', ' ', regex=True)
)

print("Columnas detectadas:")
for c in data.columns:
    print("→", c)

# ------------------------------------------------------------
# DETECCIÓN AUTOMÁTICA DE COLUMNAS
# ------------------------------------------------------------

aceite_col = [c for c in data.columns if "Oleosa" in c][0]
surf_col   = [c for c in data.columns if "Tensoactivo" in c and "Co" not in c][0]
cosurf_col = [c for c in data.columns if "tensoactivo" in c.lower() and "co" in c.lower()][0]
size_col   = [c for c in data.columns if "Tamaño" in c][0]

print("\nUsando columnas:")
print("Aceite:", aceite_col)
print("Surf:", surf_col)
print("Co-surf:", cosurf_col)
print("Tamaño:", size_col)

# ------------------------------------------------------------
# VARIABLES DE COMPOSICIÓN
# ------------------------------------------------------------

aceite = pd.to_numeric(data[aceite_col], errors="coerce")
surf   = pd.to_numeric(data[surf_col], errors="coerce")
cosurf = pd.to_numeric(data[cosurf_col], errors="coerce")

total = aceite + surf + cosurf

data['Frac_aceite'] = aceite / total
data['Frac_surf'] = surf / total
data['Frac_cosurf'] = cosurf / total
data['Smix_ratio'] = surf / cosurf

# ------------------------------------------------------------
# VARIABLE OBJETIVO (Tamaño)
# ------------------------------------------------------------

y2 = data[size_col].astype(str).str.split('±').str[0]
y2 = pd.to_numeric(y2, errors="coerce")

X2 = data[['Frac_aceite','Frac_surf','Frac_cosurf','Smix_ratio']]

datos2 = pd.concat([X2, y2], axis=1).dropna()

X2 = datos2[['Frac_aceite','Frac_surf','Frac_cosurf','Smix_ratio']]
y2 = datos2[size_col]

# ------------------------------------------------------------
# MODELO
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X2, y2, test_size=0.3, random_state=42
)

rf2 = RandomForestRegressor(n_estimators=500, random_state=42)
rf2.fit(X_train, y_train)

y_pred2 = rf2.predict(X_test)

r2_2 = r2_score(y_test, y_pred2)
rmse_2 = np.sqrt(mean_squared_error(y_test, y_pred2))

print(f"R² tamaño = {r2_2:.3f}")
print(f"RMSE tamaño = {rmse_2:.3f}")

# Importancia variables tamaño
plt.figure()
plt.bar(X2.columns, rf2.feature_importances_)
plt.title("Factores que controlan el tamaño")
plt.ylabel("Importancia")
plt.show()

# Predicción vs real tamaño
plt.figure()
plt.scatter(y_test, y_pred2)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--')
plt.xlabel("Tamaño experimental (nm)")
plt.ylabel("Predicción (nm)")
plt.title("Predicción vs Real — Tamaño")
plt.show()
 




# ============================================================
# 🏆 ÍNDICE GLOBAL DE DESEMPEÑO SNEDDS
# ============================================================

print("\n================ ÍNDICE GLOBAL SNEDDS =================")

# -------------------------
# DATOS DE TU FORMULACIÓN
# -------------------------

tamano = 40          # nm
pdi = 0.10
zeta = -15           # mV
absorcion = 0.54085
auc = 20.536

# -------------------------
# VALORES DE REFERENCIA
# -------------------------

tamano_ref = 200
pdi_ref = 0.5
zeta_ref = 30
auc_ref = 25

# -------------------------
# NORMALIZACIÓN
# -------------------------

score_tamano = 1 - (tamano / tamano_ref)
score_pdi = 1 - (pdi / pdi_ref)
score_zeta = abs(zeta) / zeta_ref
score_abs = absorcion
score_auc = auc / auc_ref

scores = [score_tamano, score_pdi, score_zeta, score_abs, score_auc]
scores = [max(0, min(1, s)) for s in scores]

# -------------------------
# ÍNDICE GLOBAL
# -------------------------

IGD = np.mean(scores)

print(f"Índice Global de Desempeño (IGD) = {IGD:.3f}")

if IGD > 0.8:
    nivel = "EXCELENTE — candidato clínico"
elif IGD > 0.6:
    nivel = "MUY BUENO — altamente prometedor"
elif IGD > 0.4:
    nivel = "ADECUADO — requiere optimización"
else:
    nivel = "BAJO desempeño"

print("Evaluación:", nivel)

# -------------------------
# GRÁFICA RADAR
# -------------------------

labels = ["Tamaño", "PDI", "Zeta", "Absorción", "AUC"]
values = scores + [scores[0]]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

plt.figure(figsize=(6,6))
ax = plt.subplot(111, polar=True)

ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title("Perfil integral del SNEDDS")
plt.show()