import matplotlib.pyplot as plt

# Etiquetas en el orden fisiológico correcto
fases = [
    "Inicial",
    "Boca",
    "Estómago",
    "Estómago → Intestino",
    "Intestino (15 min)",
    "Intestino (30 min)",
    "Intestino (45 min)",
    "Intestino (60 min)",
    "Intestino (75 min)",
    "Intestino (90 min)",
    "Intestino (105 min)",
    "Intestino (120 min)"
]

# Tamaño de partícula (nm) – TUS DATOS
tamano_nm = [
    22.85,
    29.36666667,
    25.58,
    26.29333333,
    26.80666667,
    26.80666667,
    33.41,
    26.46333333,
    30.78666667,
    32.27333333,
    26.00666667,
    37.28666667
]

plt.figure(figsize=(11, 6))
plt.plot(fases, tamano_nm, marker='o', linewidth=2)

plt.title("Evolución estructural del SNEDDS durante la digestión in vitro")
plt.xlabel("Progresión digestiva (fase fisiológica)")
plt.ylabel("Tamaño de partícula (nm)")

plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
