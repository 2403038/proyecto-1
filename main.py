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

# --- Estadística descriptiva por fase fisiológica ---
import pandas as pd
import numpy as np

# Cargar datos desde el archivo Excel antes de usar df (si no está ya cargado)
df = pd.read_excel('digestion.xlsx')

# Asumimos que las columnas relevantes son:
# 'fase' (oral, gástrica, intestinal), 'size_nm' (tamaño de partícula), 'pdi' (índice de polidispersidad)


# Estadística descriptiva solo para tamaño de partícula
# Cambia 'tamaño_particula' si tu columna tiene otro nombre
summary = df.groupby('fase').agg(
    media_tamano = ('tamaño_particula', 'mean'),
    std_tamano   = ('tamaño_particula', 'std'),
    n            = ('tamaño_particula', 'count')
).reset_index()

# Calcular coeficiente de variación después del groupby
summary['cv_tamano'] = (summary['std_tamano'] / summary['media_tamano']) * 100

# Validez experimental: n >= 3 y cv < 15% (ejemplo de criterio)
summary['validez_tamano'] = np.where((summary['n'] >= 3) & (summary['cv_tamano'] < 15), 'Válido', 'No válido')

# Redondear para presentación
summary = summary.round({
    'media_tamano': 2,
    'std_tamano': 2,
    'cv_tamano': 1
})


# Ordenar la tabla por el orden fisiológico deseado
orden_fases = ['inicial', 'boca', 'estómago', 'intestino']
summary['fase'] = summary['fase'].str.lower()  # Asegura minúsculas para comparar
summary['orden'] = summary['fase'].apply(lambda x: orden_fases.index(x) if x in orden_fases else 99)
summary = summary.sort_values('orden').drop(columns='orden')

print('\nEstadística descriptiva por fase fisiológica (tamaño de partícula):')
print(summary)

