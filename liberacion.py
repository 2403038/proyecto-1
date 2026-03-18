import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =====================================================
# 1. RUTA REAL DEL ARCHIVO (YA VERIFICADA)
# =====================================================

ruta = r"C:\Users\heber\Desktop\proyecto 1\Cinetica de liberacion.xlsx"

# =====================================================
# 2. LEER EXCEL SIN ASUMIR ENCABEZADOS
# =====================================================

df = pd.read_excel(ruta, header=None)

# =====================================================
# 3. USAR PRIMERA FILA COMO ENCABEZADO
# =====================================================

df.columns = df.iloc[0]
df = df[1:]

# =====================================================
# 4. LIMPIAR COLUMNAS IMPORTANTES
# =====================================================

df["Tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
df["%liberacion"] = pd.to_numeric(df["%liberacion"], errors="coerce")

# =====================================================
# 5. EXTRAER SOLO DATOS CINÉTICOS REALES
# (filas con tiempo Y % liberación)
# =====================================================

datos = df.dropna(subset=["Tiempo", "%liberacion"])

# Quitar tiempo = 0 (no sirve para modelo log)
datos = datos[datos["Tiempo"] > 0]

# Convertir a arrays
t = datos["Tiempo"].values.astype(float)
M = datos["%liberacion"].values.astype(float)

print("\n📊 Datos cinéticos usados:")
print(datos[["Tiempo", "%liberacion"]])

# =====================================================
# 6. MODELO KORSMEYER–PEPPAS
# Mt = Kp * t^n
# =====================================================

log_t = np.log10(t)
log_M = np.log10(M)

slope, intercept, r_value, _, _ = linregress(log_t, log_M)

n = slope
Kp = 10 ** intercept
R2 = r_value ** 2

# =====================================================
# 7. INTERPRETACIÓN MECANÍSTICA
# =====================================================

if n <= 0.45:
    mecanismo = "Difusión Fickiana"
elif n < 0.89:
    mecanismo = "Transporte anómalo"
elif abs(n - 1) < 0.1:
    mecanismo = "Relajación / erosión"
else:
    mecanismo = "Super Caso II"

# =====================================================
# 8. RESULTADOS
# =====================================================

print("\n🏆 RESULTADOS KORSMEYER–PEPPAS")
print("n =", n)
print("Kp =", Kp)
print("R² =", R2)
print("Mecanismo:", mecanismo)

# =====================================================
# 9. GRÁFICA PROFESIONAL
# =====================================================

plt.figure(figsize=(7,5))

plt.scatter(log_t, log_M, label="Datos experimentales")

plt.plot(
    log_t,
    intercept + slope * log_t,
    label=f"Ajuste KP\nn={n:.3f}   R²={R2:.3f}"
)

plt.xlabel("log Tiempo (h)")
plt.ylabel("log % liberación")
plt.title("Modelo Korsmeyer–Peppas — Liberación de CPT (SNEDDS)")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =====================================================
# 1. RUTA DEL ARCHIVO
# =====================================================

ruta = r"C:\Users\heber\Desktop\proyecto 1\Cinetica de liberacion.xlsx"

# =====================================================
# 2. LEER EXCEL
# =====================================================

df = pd.read_excel(ruta, header=None)

df.columns = df.iloc[0]
df = df[1:]

# Limpiar datos
df["Tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
df["%liberacion"] = pd.to_numeric(df["%liberacion"], errors="coerce")

datos = df.dropna(subset=["Tiempo", "%liberacion"])
datos = datos[datos["Tiempo"] > 0]

t = datos["Tiempo"].values.astype(float)
M = datos["%liberacion"].values.astype(float)

print("\n📊 Datos usados:")
print(datos[["Tiempo", "%liberacion"]])

# =====================================================
# MODELO 1 — KORSMEYER–PEPPAS
# =====================================================

log_t = np.log10(t)
log_M = np.log10(M)

slope_kp, intercept_kp, r_kp, _, _ = linregress(log_t, log_M)

n = slope_kp
Kp = 10 ** intercept_kp
R2_kp = r_kp ** 2

# =====================================================
# MODELO 2 — HIGUCHI
# Mt = k * sqrt(t)
# =====================================================

sqrt_t = np.sqrt(t)

slope_h, intercept_h, r_h, _, _ = linregress(sqrt_t, M)

Kh = slope_h
R2_h = r_h ** 2

# =====================================================
# MODELO 3 — PRIMER ORDEN
# ln(100 − Mt) vs t
# =====================================================

M_remaining = 100 - M
mask = M_remaining > 0

t_po = t[mask]
M_po = M_remaining[mask]

ln_M = np.log(M_po)

slope_po, intercept_po, r_po, _, _ = linregress(t_po, ln_M)

K1 = -slope_po
R2_po = r_po ** 2

# =====================================================
# MODELO 4 — WEIBULL
# ln(-ln(1 - Mt/100)) vs ln(t)
# =====================================================

F = M / 100
mask = (F > 0) & (F < 1)

t_w = t[mask]
F_w = F[mask]

Y = np.log(-np.log(1 - F_w))
X = np.log(t_w)

slope_w, intercept_w, r_w, _, _ = linregress(X, Y)

beta = slope_w
alpha = np.exp(-intercept_w / slope_w)
R2_w = r_w ** 2

# =====================================================
# RESULTADOS
# =====================================================

print("\n🏆 RESULTADOS MODELOS")

print("\nKorsmeyer–Peppas")
print("n =", n, "Kp =", Kp, "R² =", R2_kp)

print("\nHiguchi")
print("Kh =", Kh, "R² =", R2_h)

print("\nPrimer orden")
print("K1 =", K1, "R² =", R2_po)

print("\nWeibull")
print("beta =", beta, "alpha =", alpha, "R² =", R2_w)

# =====================================================
# MEJOR MODELO
# =====================================================

modelos = {
    "Korsmeyer–Peppas": R2_kp,
    "Higuchi": R2_h,
    "Primer orden": R2_po,
    "Weibull": R2_w
}

mejor = max(modelos, key=modelos.get)

print("\n🥇 MEJOR MODELO:", mejor)

# =====================================================
# GRAFICA LIBERACIÓN CLÁSICA
# =====================================================

plt.figure(figsize=(7,5))

plt.plot(t, M, 'o-', label="Liberación experimental")

plt.xlabel("Tiempo")
plt.ylabel("% liberación")
plt.title("Perfil de liberación de CPT — SNEDDS")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =====================================================
# 1. RUTA DEL ARCHIVO
# =====================================================

ruta = r"C:\Users\heber\Desktop\proyecto 1\Cinetica de liberacion.xlsx"

# =====================================================
# 2. LEER Y LIMPIAR EXCEL
# =====================================================

df = pd.read_excel(ruta, header=None)

df.columns = df.iloc[0]
df = df[1:]

df["Tiempo"] = pd.to_numeric(df["Tiempo"], errors="coerce")
df["%liberacion"] = pd.to_numeric(df["%liberacion"], errors="coerce")

datos = df.dropna(subset=["Tiempo", "%liberacion"])
datos = datos[datos["Tiempo"] > 0]

t = datos["Tiempo"].values.astype(float)
M = datos["%liberacion"].values.astype(float)

# =====================================================
# MODELO 1 — KORSMEYER–PEPPAS
# =====================================================

log_t = np.log10(t)
log_M = np.log10(M)

slope_kp, intercept_kp, r_kp, _, _ = linregress(log_t, log_M)
R2_kp = r_kp**2

plt.figure()
plt.scatter(log_t, log_M)
plt.plot(log_t, intercept_kp + slope_kp * log_t)
plt.title(f"Korsmeyer–Peppas\nR² = {R2_kp:.3f}")
plt.xlabel("log Tiempo")
plt.ylabel("log % liberación")
plt.grid(True)

# =====================================================
# MODELO 2 — HIGUCHI
# =====================================================

sqrt_t = np.sqrt(t)

slope_h, intercept_h, r_h, _, _ = linregress(sqrt_t, M)
R2_h = r_h**2

plt.figure()
plt.scatter(sqrt_t, M)
plt.plot(sqrt_t, intercept_h + slope_h * sqrt_t)
plt.title(f"Higuchi\nR² = {R2_h:.3f}")
plt.xlabel("√Tiempo")
plt.ylabel("% liberación")
plt.grid(True)

# =====================================================
# MODELO 3 — PRIMER ORDEN
# =====================================================

M_remaining = 100 - M
mask = M_remaining > 0

t_po = t[mask]
ln_M = np.log(M_remaining[mask])

slope_po, intercept_po, r_po, _, _ = linregress(t_po, ln_M)
R2_po = r_po**2

plt.figure()
plt.scatter(t_po, ln_M)
plt.plot(t_po, intercept_po + slope_po * t_po)
plt.title(f"Primer orden\nR² = {R2_po:.3f}")
plt.xlabel("Tiempo")
plt.ylabel("ln(100 − % liberación)")
plt.grid(True)

# =====================================================
# MODELO 4 — WEIBULL
# =====================================================

F = M / 100
mask = (F > 0) & (F < 1)

t_w = t[mask]
F_w = F[mask]

X = np.log(t_w)
Y = np.log(-np.log(1 - F_w))

slope_w, intercept_w, r_w, _, _ = linregress(X, Y)
R2_w = r_w**2

plt.figure()
plt.scatter(X, Y)
plt.plot(X, intercept_w + slope_w * X)
plt.title(f"Weibull\nR² = {R2_w:.3f}")
plt.xlabel("ln Tiempo")
plt.ylabel("ln(-ln(1 − F))")
plt.grid(True)

# =====================================================
# MOSTRAR TODAS LAS FIGURAS
# =====================================================

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =====================================================
# 1. CARGAR DATOS
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

# =====================================================
# MODELO 1 — KORSMEYER–PEPPAS
# =====================================================

log_t = np.log10(t)
log_M = np.log10(M)

slope_kp, intercept_kp, _, _, _ = linregress(log_t, log_M)

n = slope_kp
Kp = 10**intercept_kp

M_kp = Kp * t**n

# =====================================================
# MODELO 2 — HIGUCHI
# =====================================================

sqrt_t = np.sqrt(t)

slope_h, intercept_h, _, _, _ = linregress(sqrt_t, M)

Kh = slope_h
M_h = Kh * np.sqrt(t) + intercept_h

# =====================================================
# MODELO 3 — PRIMER ORDEN
# =====================================================

M_remaining = 100 - M
mask = M_remaining > 0

t_po = t[mask]
ln_M = np.log(M_remaining[mask])

slope_po, intercept_po, _, _, _ = linregress(t_po, ln_M)

K1 = -slope_po

M_po = 100 * (1 - np.exp(-K1 * t))

# =====================================================
# MODELO 4 — WEIBULL
# =====================================================

F = M / 100
mask = (F > 0) & (F < 1)

t_w = t[mask]
F_w = F[mask]

X = np.log(t_w)
Y = np.log(-np.log(1 - F_w))

slope_w, intercept_w, _, _, _ = linregress(X, Y)

beta = slope_w
alpha = np.exp(-intercept_w / slope_w)

M_w = 100 * (1 - np.exp(-(t / alpha) ** beta))

# =====================================================
# GRAFICA ÚNICA — DATOS + MODELOS
# =====================================================

plt.figure(figsize=(9,6))

plt.scatter(t, M, color="black", s=70,
            label="Datos experimentales")

plt.plot(t, M_kp, label="Korsmeyer–Peppas")
plt.plot(t, M_h, label="Higuchi")
plt.plot(t, M_po, label="Primer orden")
plt.plot(t, M_w, label="Weibull")

plt.xlabel("Tiempo (h)", fontsize=12)
plt.ylabel("% liberación", fontsize=12)
plt.title("Perfil de liberación de CPT — Ajuste de modelos", fontsize=14)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

















