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
# 4. FILTRAR FASE INTESTINAL
# =========================

df_int = df[
    df[col_fase].str.contains("Intestino", case=False, na=False)
].copy()

df_int[col_tamano] = pd.to_numeric(df_int[col_tamano], errors="coerce")
df_int[col_tiempo] = pd.to_numeric(df_int[col_tiempo], errors="coerce")

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
# 6. VARIABLES
# =========================

t = datos[col_tiempo]
tam_prom = datos['mean']
tam_std = datos['std']

# =========================
# 7. CONFIGURACIÓN DE ESTILO
# =========================

plt.style.use("seaborn-v0_8-whitegrid")

plt.figure(figsize=(7,5))

# =========================
# 8. GRAFICA
# =========================

plt.errorbar(
    t,
    tam_prom,
    yerr=tam_std,
    fmt='o',
    color='navy',
    ecolor='black',
    capsize=4,
    markersize=7,
    label="Mean ± SD"
)

plt.plot(
    t,
    tam_prom,
    color='navy',
    linewidth=2
)

# =========================
# 9. ETIQUETAS
# =========================

plt.xlabel("Tiempo de digestión (min)", fontsize=12)
plt.ylabel("Tamaño de partícula (nm)", fontsize=12)

plt.title("Estabilidad del SNEDDS durante digestión intestinal", fontsize=13)

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.legend()

# =========================
# 10. GUARDAR FIGURA
# =========================

plt.tight_layout()

plt.savefig(
    "grafica_digestión_intestinal.png",
    dpi=300
)

plt.show()





# =========================
# 11. VELOCIDAD DE CAMBIO
# =========================

D = datos['mean'].values
t = datos['Tiempo_min'].values

velocidad = np.diff(D) / np.diff(t)

tiempo_vel = (t[:-1] + t[1:]) / 2

df_vel = pd.DataFrame({
    "Tiempo_min": tiempo_vel,
    "dD_dt (nm/min)": velocidad
})

print("\nVelocidad de cambio del tamaño:")
print(df_vel)
# =========================
# 12. GRAFICA VELOCIDAD
# =========================

plt.figure(figsize=(7,5))

plt.plot(
    tiempo_vel,
    velocidad,
    marker='o',
    linewidth=2
)

plt.axhline(0)

plt.xlabel("Tiempo de digestión (min)")
plt.ylabel("Velocidad de cambio (nm/min)")
plt.title("Cinética de cambio del tamaño de partícula")

plt.grid(True)

plt.show()










import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# CARGAR EXCEL
# =========================

df = pd.read_excel("Datos generales digestion.xlsx")

# =========================
# LIMPIAR NOMBRES
# =========================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

# =========================
# DEFINIR COLUMNAS
# =========================

col_fase = "Fase"
col_tamano = "Tamaño_de_particula"
col_tiempo = "Tiempo_min"

# =========================
# LIMPIAR DATOS
# =========================

df[col_tamano] = pd.to_numeric(df[col_tamano], errors="coerce")
df[col_tiempo] = pd.to_numeric(df[col_tiempo], errors="coerce")

df = df.dropna(subset=[col_tamano, col_tiempo])

# =========================
# PROMEDIO Y SD
# =========================

datos = (
    df
    .groupby([col_fase, col_tiempo])[col_tamano]
    .agg(['mean','std'])
    .reset_index()
)

datos = datos.sort_values(col_tiempo)

t = datos[col_tiempo].values
tam_prom = datos['mean'].values
tam_std = datos['std'].values

# =========================
# MODELO CIENTÍFICO
# =========================

def modelo(t,a,b,c):
    return a*t**2 + b*t + c

params,_ = curve_fit(modelo,t,tam_prom)

a,b,c = params

# predicción del modelo
t_fit = np.linspace(min(t),max(t),300)
tam_fit = modelo(t_fit,a,b,c)

# =========================
# CALCULO R2
# =========================

tam_pred = modelo(t,a,b,c)

ss_res = np.sum((tam_prom - tam_pred)**2)
ss_tot = np.sum((tam_prom - np.mean(tam_prom))**2)

r2 = 1 - ss_res/ss_tot

print("\nModelo cinético:")
print("a =",a)
print("b =",b)
print("c =",c)
print("R2 =",r2)

# =========================
# GRAFICA PROFESIONAL
# =========================

plt.figure(figsize=(9,5))

# zonas digestivas
plt.axvspan(0,5,color="#FFF3B0",alpha=0.6,label="Boca")
plt.axvspan(5,120,color="#FFB3B3",alpha=0.4,label="Estómago")
plt.axvspan(120,240,color="#B7E4C7",alpha=0.4,label="Intestino")

# datos experimentales
plt.errorbar(
    t,
    tam_prom,
    yerr=tam_std,
    fmt='o',
    color="#1f77b4",
    ecolor="black",
    capsize=4,
    markersize=7,
    label="Datos experimentales ± SD"
)

# linea de datos
plt.plot(
    t,
    tam_prom,
    color="#1f77b4",
    linewidth=1.5
)

# modelo cinético
plt.plot(
    t_fit,
    tam_fit,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Modelo cinético (R² = {r2:.3f})"
)

# etiquetas
plt.xlabel("Tiempo de digestión (min)",fontsize=12)
plt.ylabel("Tamaño de partícula (nm)",fontsize=12)

plt.title(
"Evolución del tamaño de partícula del SNEDDS durante digestión gastrointestinal",
fontsize=13
)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

# guardar figura
plt.savefig(
"grafica_SNEDDS_digestión_modelo.png",
dpi=600
)

plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# CARGAR DATOS
# =========================

df = pd.read_excel("Datos generales digestion.xlsx")

df.columns = df.columns.str.strip().str.replace(" ", "_")

col_fase = "Fase"
col_tamano = "Tamaño_de_particula"
col_tiempo = "Tiempo_min"

df[col_tamano] = pd.to_numeric(df[col_tamano], errors="coerce")
df[col_tiempo] = pd.to_numeric(df[col_tiempo], errors="coerce")

df = df.dropna(subset=[col_tamano, col_tiempo])

# =========================
# PROMEDIO Y SD
# =========================

datos = (
    df.groupby([col_fase, col_tiempo])[col_tamano]
    .agg(['mean','std'])
    .reset_index()
)

datos = datos.sort_values(col_tiempo)

t = datos[col_tiempo].values
tam_prom = datos['mean'].values
tam_std = datos['std'].values

# =========================
# MODELO
# =========================

def modelo(t,a,b,c):
    return a*t**2 + b*t + c

params,_ = curve_fit(modelo,t,tam_prom)

a,b,c = params

t_fit = np.linspace(min(t),max(t),300)
tam_fit = modelo(t_fit,a,b,c)

# R2
tam_pred = modelo(t,a,b,c)

ss_res = np.sum((tam_prom - tam_pred)**2)
ss_tot = np.sum((tam_prom - np.mean(tam_prom))**2)

r2 = 1 - ss_res/ss_tot

# =========================
# VELOCIDAD dD/dt
# =========================

vel = np.diff(tam_prom)/np.diff(t)
t_vel = (t[:-1] + t[1:])/2

# =========================
# FIGURA DOS PANELES
# =========================

fig,ax = plt.subplots(2,1,figsize=(9,8))

# PANEL A
ax[0].axvspan(0,5,color="#FFF3B0",alpha=0.6)
ax[0].axvspan(5,120,color="#FFB3B3",alpha=0.4)
ax[0].axvspan(120,240,color="#B7E4C7",alpha=0.4)

ax[0].errorbar(t,tam_prom,yerr=tam_std,fmt='o',capsize=4,label="Datos ± SD")

ax[0].plot(t,tam_prom)

ax[0].plot(t_fit,tam_fit,'--',label=f"Modelo (R²={r2:.3f})")

ax[0].set_ylabel("Tamaño de partícula (nm)")
ax[0].set_title("A) Evolución del tamaño de partícula")

ax[0].legend()
ax[0].grid(alpha=0.3)

# PANEL B

ax[1].plot(t_vel,vel,'o-',color="black")

ax[1].axhline(0)

ax[1].set_xlabel("Tiempo de digestión (min)")
ax[1].set_ylabel("dD/dt (nm/min)")
ax[1].set_title("B) Velocidad de cambio del tamaño")

ax[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig("figura_SNEDDS_cinetica.png",dpi=600)

plt.show()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# CARGAR DATOS
# =========================

df = pd.read_excel("Datos generales digestion.xlsx")

df.columns = df.columns.str.strip().str.replace(" ", "_")

col_fase = "Fase"
col_tamano = "Tamaño_de_particula"
col_tiempo = "Tiempo_min"

df[col_tamano] = pd.to_numeric(df[col_tamano], errors="coerce")
df[col_tiempo] = pd.to_numeric(df[col_tiempo], errors="coerce")

df = df.dropna(subset=[col_tamano, col_tiempo])

# =========================
# PROMEDIOS
# =========================

datos = (
    df.groupby([col_fase, col_tiempo])[col_tamano]
    .agg(['mean','std'])
    .reset_index()
)

datos = datos.sort_values(col_tiempo)

t = datos[col_tiempo].values
tam_prom = datos['mean'].values
tam_std = datos['std'].values

# =========================
# MODELO EXPONENCIAL
# =========================

def modelo_exp(t,Dinf,D0,k):
    return Dinf + (D0 - Dinf)*np.exp(-k*t)

p0 = [26,25,0.01]

params,_ = curve_fit(modelo_exp,t,tam_prom,p0=p0,maxfev=10000)

Dinf,D0,k = params

t_fit = np.linspace(min(t),max(t),300)

tam_fit = modelo_exp(t_fit,Dinf,D0,k)

# =========================
# R2
# =========================

tam_pred = modelo_exp(t,Dinf,D0,k)

ss_res = np.sum((tam_prom - tam_pred)**2)
ss_tot = np.sum((tam_prom - np.mean(tam_prom))**2)

r2 = 1 - ss_res/ss_tot

print("\nModelo exponencial")
print("D∞ =",Dinf)
print("D0 =",D0)
print("k =",k)
print("R2 =",r2)

# =========================
# VELOCIDAD
# =========================

vel = np.diff(tam_prom)/np.diff(t)
t_vel = (t[:-1] + t[1:])/2

# =========================
# FIGURA DOS PANELES
# =========================

fig,ax = plt.subplots(2,1,figsize=(9,8))

# PANEL A
ax[0].axvspan(0,5,color="#FFF3B0",alpha=0.6,label="Boca")
ax[0].axvspan(5,120,color="#FFB3B3",alpha=0.4,label="Estómago")
ax[0].axvspan(120,240,color="#B7E4C7",alpha=0.4,label="Intestino")

ax[0].errorbar(t,tam_prom,yerr=tam_std,fmt='o',capsize=4,label="Datos ± SD")

ax[0].plot(t,tam_prom)

ax[0].plot(t_fit,tam_fit,'--',linewidth=2,label=f"Modelo exponencial (R²={r2:.3f})")

ax[0].set_ylabel("Tamaño de partícula (nm)")
ax[0].set_title("A) Evolución del tamaño de partícula del SNEDDS")

ax[0].legend()
ax[0].grid(alpha=0.3)

# PANEL B
ax[1].plot(t_vel,vel,'o-',color="black")

ax[1].axhline(0)

ax[1].set_xlabel("Tiempo de digestión (min)")
ax[1].set_ylabel("dD/dt (nm/min)")
ax[1].set_title("B) Velocidad de cambio")

ax[1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig("SNEDDS_modelo_exponencial.png",dpi=600)

plt.show()







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# CARGAR DATOS
# =========================

df = pd.read_excel("Datos generales digestion.xlsx")

df.columns = df.columns.str.strip().str.replace(" ", "_")

col_fase = "Fase"
col_tamano = "Tamaño_de_particula"
col_tiempo = "Tiempo_min"

df[col_tamano] = pd.to_numeric(df[col_tamano], errors="coerce")
df[col_tiempo] = pd.to_numeric(df[col_tiempo], errors="coerce")

df = df.dropna(subset=[col_tamano, col_tiempo])

# =========================
# PROMEDIO
# =========================

datos = (
    df.groupby([col_fase, col_tiempo])[col_tamano]
    .mean()
    .reset_index()
)

datos = datos.sort_values(col_tiempo)

t = datos[col_tiempo].values
D = datos['Tamaño_de_particula'].values

# =========================
# FUNCION R2
# =========================

def calc_r2(y, y_pred):
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res/ss_tot

# =========================
# 1. MODELO EXPONENCIAL
# =========================

def modelo_exp(t, Dinf, D0, k):
    return Dinf + (D0 - Dinf)*np.exp(-k*t)

p0_exp = [D[-1], D[0], 0.01]
par_exp,_ = curve_fit(modelo_exp, t, D, p0=p0_exp, maxfev=10000)

D_exp = modelo_exp(t, *par_exp)
r2_exp = calc_r2(D, D_exp)

# =========================
# 2. MODELO LOGÍSTICO
# =========================

def modelo_log(t, Dmax, k, t0):
    return Dmax / (1 + np.exp(-k*(t - t0)))

p0_log = [max(D), 0.01, np.mean(t)]
par_log,_ = curve_fit(modelo_log, t, D, p0=p0_log, maxfev=10000)

D_log = modelo_log(t, *par_log)
r2_log = calc_r2(D, D_log)

# =========================
# 3. POLINOMIAL 2° ORDEN
# =========================

coef_poly = np.polyfit(t, D, 2)
D_poly = np.polyval(coef_poly, t)
r2_poly = calc_r2(D, D_poly)

# =========================
# 4. MODELO LINEAL
# =========================

coef_lin = np.polyfit(t, D, 1)
D_lin = np.polyval(coef_lin, t)
r2_lin = calc_r2(D, D_lin)

# =========================
# RESULTADOS
# =========================

print("\n=== R² DE LOS MODELOS ===")
print("Exponencial:", r2_exp)
print("Logístico:", r2_log)
print("Polinomial:", r2_poly)
print("Lineal:", r2_lin)

# =========================
# CURVAS SUAVES
# =========================

t_fit = np.linspace(min(t), max(t), 300)

plt.figure(figsize=(9,6))

# datos experimentales
plt.scatter(t, D, color="black", label="Datos")

# curvas
plt.plot(t_fit, modelo_exp(t_fit, *par_exp),
         label=f"Exponencial (R²={r2_exp:.3f})")

plt.plot(t_fit, modelo_log(t_fit, *par_log),
         label=f"Logístico (R²={r2_log:.3f})")

plt.plot(t_fit, np.polyval(coef_poly, t_fit),
         label=f"Polinomial (R²={r2_poly:.3f})")

plt.plot(t_fit, np.polyval(coef_lin, t_fit),
         label=f"Lineal (R²={r2_lin:.3f})")

plt.xlabel("Tiempo (min)")
plt.ylabel("Tamaño de partícula (nm)")
plt.title("Comparación de modelos cinéticos")

plt.legend()
plt.grid(True)

plt.show()