import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
df = pd.read_csv("digestion.csv")

df.columns = df.columns.str.lower().str.strip()
boca = df[df["fase"] == "boca"]
estomago = df[df["fase"] == "estomago"]
intestino = df[df["fase"] == "intestino"]

def modelo_estomago(t, Deq, k):
    D0 = estomago["tamaño_nm"].iloc[0]
    return Deq + (D0 - Deq) * np.exp(-k * t)

def modelo_intestino(t, a, b):
    return a * t + b

popt_boca, _ = curve_fit(
    modelo_boca,
    boca["tiempo_h"],
    boca["tamaño_nm"]
)
popt_estomago, _ = curve_fit(
    modelo_estomago,
    estomago["tiempo_h"],
    estomago["tamaño_nm"]
)
popt_intestino, _ = curve_fit(
    modelo_intestino,
    intestino["tiempo_h"],
    intestino["tamaño_nm"]
)
plt.figure(figsize=(8,5))

plt.scatter(df["tiempo_h"], df["tamaño_nm"], label="Experimental", color="black")

t_b = np.linspace(boca["tiempo_h"].min(), boca["tiempo_h"].max(), 100)
plt.plot(t_b, modelo_boca(t_b, *popt_boca), label="Modelo Boca")

t_e = np.linspace(estomago["tiempo_h"].min(), estomago["tiempo_h"].max(), 100)
plt.plot(t_e, modelo_estomago(t_e, *popt_estomago), label="Modelo Estómago")

t_i = np.linspace(intestino["tiempo_h"].min(), intestino["tiempo_h"].max(), 100)
plt.plot(t_i, modelo_intestino(t_i, *popt_intestino), label="Modelo Intestino")

plt.xlabel("Tiempo (h)")
plt.ylabel("Tamaño (nm)")
plt.legend()
plt.show()
