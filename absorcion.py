# ============================================================
# 🧠 MODELO INTEGRAL SNEDDS PEGILADO
# Distribución sistémica + dinámica intracelular
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


print("\n================ DISTRIBUCIÓN SISTÉMICA (PEG) =================")

# ============================================================
# 🩸 PARTE 1 — DISTRIBUCIÓN SISTÉMICA
# Plasma → Tejido → Intracelular
# ============================================================

# Parámetros (PEG stealth)

Fa = 0.54      # fracción absorbida desde intestino
Ka = 1.0       # absorción al plasma

Ke = 0.25      # eliminación plasmática

Kpt = 0.6      # plasma → tejido
Ktp = 0.4      # tejido → plasma

Ktc = 0.3      # tejido → célula
Kct = 0.2      # célula → tejido

Kdeg = 0.1     # degradación intracelular


# -------------------------
# MODELO SISTÉMICO
# -------------------------

def modelo_sistemico(y, t):
    Cp, Ct, Cc = y
    
    dCp = Ka*Fa - Ke*Cp - Kpt*Cp + Ktp*Ct
    dCt = Kpt*Cp - Ktp*Ct - Ktc*Ct + Kct*Cc
    dCc = Ktc*Ct - Kct*Cc - Kdeg*Cc
    
    return [dCp, dCt, dCc]


# -------------------------
# SIMULACIÓN
# -------------------------

y0_sys = [0, 0, 0]
t = np.linspace(0, 24, 200)

sol_sys = odeint(modelo_sistemico, y0_sys, t)

Cp = sol_sys[:, 0]
Ct = sol_sys[:, 1]
Cc = sol_sys[:, 2]


# -------------------------
# GRÁFICA SISTÉMICA
# -------------------------

plt.figure(figsize=(8, 5))
plt.plot(t, Cp, label="Plasma")
plt.plot(t, Ct, label="Tejido")
plt.plot(t, Cc, label="Intracelular")

plt.xlabel("Tiempo (h)")
plt.ylabel("Concentración relativa")
plt.title("Distribución sistémica del SNEDDS PEGilado")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------
# MÉTRICAS SISTÉMICAS
# -------------------------

print(f"Cmax plasmática = {Cp.max():.3f}")
print(f"Cmax intracelular = {Cc.max():.3f}")
print(f"Tmax celular ≈ {t[np.argmax(Cc)]:.2f} h")


# ============================================================
# 🧬 PARTE 2 — DINÁMICA INTRACELULAR
# SNEDDS → Fármaco libre → Metabolizado
# ============================================================

print("\n================ DINÁMICA INTRACELULAR =================")

Cext = Cc.max()   # concentración extracelular

Kin = 0.6         # endocitosis
Kdis = 0.5        # desintegración SNEDDS
Kexo = 0.1        # exocitosis

Kmet = 0.3        # metabolismo
Kefflux = 0.2     # salida del fármaco


# -------------------------
# MODELO INTRACELULAR
# -------------------------

def modelo_intracelular(y, t):
    N, D, M = y
    
    dN = Kin*Cext - Kdis*N - Kexo*N
    dD = Kdis*N - Kmet*D - Kefflux*D
    dM = Kmet*D
    
    return [dN, dD, dM]


# -------------------------
# SIMULACIÓN
# -------------------------

y0_cell = [0, 0, 0]

sol_cell = odeint(modelo_intracelular, y0_cell, t)

N = sol_cell[:, 0]
D = sol_cell[:, 1]
M = sol_cell[:, 2]


# -------------------------
# GRÁFICA INTRACELULAR
# -------------------------

plt.figure(figsize=(8, 5))
plt.plot(t, N, label="SNEDDS intracelular")
plt.plot(t, D, label="Fármaco libre")
plt.plot(t, M, label="Metabolizado")

plt.xlabel("Tiempo (h)")
plt.ylabel("Cantidad relativa")
plt.title("Destino intracelular del SNEDDS PEGilado")
plt.legend()
plt.grid(True)
plt.show()


# -------------------------
# MÉTRICAS INTRACELULARES
# -------------------------

print(f"Máximo SNEDDS intracelular = {N.max():.3f}")
print(f"Máxima concentración fármaco libre = {D.max():.3f}")
print(f"Tiempo a máximo fármaco ≈ {t[np.argmax(D)]:.2f} h")