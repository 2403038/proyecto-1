# =========================
# IMPORTACIONES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARTE 1: GRÁFICA DE DIGESTIÓN (PROMEDIOS EXPERIMENTALES)
# =========================================================

# Ver hojas disponibles
excel_file = pd.ExcelFile('digestion.xlsx')
print('Hojas disponibles en digestion.xlsx:', excel_file.sheet_names)

# Fases en orden fisiológico
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

# Tamaño promedio (nm)
tamano_nm = [
    22.85,
    29.37,
    25.58,
    26.29,
    26.81,
    26.81,
    33.41,
    26.46,
    30.79,
    32.27,
    26.01,
    37.29
]

# Gráfica
plt.figure(figsize=(11, 6))
plt.plot(fases, tamano_nm, marker='o', linewidth=2)

plt.title("Evolución estructural del SNEDDS durante la digestión in vitro")
plt.xlabel("Progresión digestiva (fase fisiológica)")
plt.ylabel("Tamaño de partícula (nm)")

plt.xticks(rotation=45, ha="right")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# =========================================================
# PARTE 2: ESTADÍSTICA DESCRIPTIVA (DATOS CRUDOS)
# =========================================================

# Ver hojas disponibles
excel_stats = pd.ExcelFile('Datos generales digestion.xlsx')
print('Hojas disponibles en Datos generales digestion.xlsx:', excel_stats.sheet_names)

# Cargar datos
df = pd.read_excel('Datos generales digestion.xlsx')

# ---------------------------------------------------------
# LIMPIEZA DE COLUMNAS
# ---------------------------------------------------------
df.columns = df.columns.str.strip()

print("\nColumnas limpias:")
print(df.columns)

# Renombrar columnas relevantes
df = df.rename(columns={
    "Fase": "fase",
    "Tamaño de particula": "tamano"
})

# ---------------------------------------------------------
# LIMPIEZA DE DATOS
# ---------------------------------------------------------
df["fase"] = df["fase"].astype(str).str.strip()

# Corregir errores tipográficos
df["fase"] = df["fase"].replace({
    "Incial": "Inicial"
})

# Normalizar para análisis
df["fase"] = df["fase"].str.lower()

# ---------------------------------------------------------
# MAPEO A FASES FISIOLÓGICAS GENERALES
# ---------------------------------------------------------
def clasificar_fase(f):
    if "inicial" in f:
        return "Inicial"
    elif "boca" in f:
        return "Boca"
    elif "estómago" in f or "estomago" in f:
        return "Estómago"
    elif "intestino" in f:
        return "Intestino"
    else:
        return "Otra"

df["fase_fisiologica"] = df["fase"].apply(clasificar_fase)

# ---------------------------------------------------------
# ESTADÍSTICA DESCRIPTIVA
# ---------------------------------------------------------
summary = (
    df
    .groupby("fase_fisiologica")["tamano"]
    .agg(
        media_tamano="mean",
        std_tamano="std",
        n="count"
    )
    .reset_index()
)

# Coeficiente de variación
summary["cv_tamano"] = (summary["std_tamano"] / summary["media_tamano"]) * 100

# Criterio de validez
summary["validez_tamano"] = np.where(
    (summary["n"] >= 3) & (summary["cv_tamano"] < 15),
    "Válido",
    "No válido"
)

# Redondeo
summary = summary.round({
    "media_tamano": 2,
    "std_tamano": 2,
    "cv_tamano": 1
})

# ---------------------------------------------------------
# ORDEN FISIOLÓGICO FINAL
# ---------------------------------------------------------
orden = ["Inicial", "Boca", "Estómago", "Intestino"]
summary["fase_fisiologica"] = pd.Categorical(
    summary["fase_fisiologica"],
    categories=orden,
    ordered=True
)

summary = summary.sort_values("fase_fisiologica")

# ---------------------------------------------------------
# MOSTRAR Y GUARDAR RESULTADOS
# ---------------------------------------------------------
print("\n=====================================")
print("ESTADÍSTICA DESCRIPTIVA — TAMAÑO")
print("=====================================")
print(summary)

summary.to_excel(
    "tabla_estadistica_tamano_SNEDDS.xlsx",
    index=False
)