import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# ETAPAS EXPERIMENTALES
# =========================

etapas = [
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

# =========================
# DATOS
# =========================

tamano = [
    22.86,
    29.37,
    25.94,
    26.30,
    26.80,
    26.80,
    33.40,
    26.50,
    30.80,
    32.20,
    26.00,
    37.30
]

pdi = [
    0.122,
    0.303666667,
    0.140666667,
    0.246,
    0.140333333,
    0.143666667,
    0.111,
    0.179333333,
    0.134333333,
    0.134333333,
    0.150333333,
    0.152333333
]

# =========================
# CREAR DATAFRAME
# =========================

df = pd.DataFrame({
    "fase_fisiologica": etapas,
    "tamano": tamano,
    "PDI": pdi
})

# =========================
# FUNCIÓN ESTADÍSTICA
# =========================

def estadistica(variable, nombre_variable):
    media = np.mean(variable)
    std = np.std(variable, ddof=1)
    n = len(variable)
    cv = (std / media) * 100
    
    validez = "Válido" if cv <= 20 else "No válido"
    
    tabla = pd.DataFrame({
        "Variable": [nombre_variable],
        "Media": [round(media,3)],
        "Desv_std": [round(std,3)],
        "n": [n],
        "CV (%)": [round(cv,2)],
        "Validez": [validez]
    })
    
    return tabla

# =========================
# TABLA ESTADÍSTICA TAMAÑO
# =========================

tabla_tamano = estadistica(tamano, "Tamaño de partícula (nm)")

print("\n=====================================")
print("ESTADÍSTICA DESCRIPTIVA — TAMAÑO")
print("=====================================")
print(tabla_tamano)

# =========================
# TABLA ESTADÍSTICA PDI
# =========================

tabla_pdi = estadistica(pdi, "PDI")

print("\n=====================================")
print("ESTADÍSTICA DESCRIPTIVA — PDI")
print("=====================================")
print(tabla_pdi)

# =========================
# GRÁFICA TAMAÑO
# =========================

plt.figure(figsize=(10,6))
plt.plot(etapas, tamano, marker='o')

plt.title("Evolución estructural del SNEDDS durante la digestión in vitro")
plt.xlabel("Progresión digestiva (fase fisiológica)")
plt.ylabel("Tamaño de partícula (nm)")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# GRÁFICA PDI
# =========================

plt.figure(figsize=(10,6))
plt.plot(etapas, pdi, marker='o')

plt.title("Evolución del PDI del SNEDDS durante la digestión in vitro")
plt.xlabel("Progresión digestiva (fase fisiológica)")
plt.ylabel("PDI")

plt.xticks(rotation=45)
plt.grid(True)
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

import pandas as pd
import numpy as np

# =========================================================
# PARTE 2: ESTADÍSTICA DESCRIPTIVA (PDI – DATOS CRUDOS)
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

# ---------------------------------------------------------
# RENOMBRAR COLUMNAS RELEVANTES
# ⚠️ AJUSTA EL NOMBRE EXACTO DE PDI SEGÚN TU EXCEL
# ---------------------------------------------------------
df = df.rename(columns={
    "Fase": "fase",
    "Indice de polidispersidad": "pdi"   # ← cambia si tu Excel usa otro nombre
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
# ESTADÍSTICA DESCRIPTIVA — PDI
# ---------------------------------------------------------
summary_pdi = (
    df
    .groupby("fase_fisiologica")["pdi"]
    .agg(
        media_pdi="mean",
        std_pdi="std",
        n="count"
    )
    .reset_index()
)

# Coeficiente de variación
summary_pdi["cv_pdi"] = (summary_pdi["std_pdi"] / summary_pdi["media_pdi"]) * 100

# Criterio de validez
summary_pdi["validez_pdi"] = np.where(
    (summary_pdi["n"] >= 3) & (summary_pdi["cv_pdi"] < 15),
    "Válido",
    "No válido"
)

# Redondeo
summary_pdi = summary_pdi.round({
    "media_pdi": 3,
    "std_pdi": 3,
    "cv_pdi": 1
})

# ---------------------------------------------------------
# ORDEN FISIOLÓGICO FINAL
# ---------------------------------------------------------
orden = ["Inicial", "Boca", "Estómago", "Intestino"]
summary_pdi["fase_fisiologica"] = pd.Categorical(
    summary_pdi["fase_fisiologica"],
    categories=orden,
    ordered=True
)

summary_pdi = summary_pdi.sort_values("fase_fisiologica")

# ---------------------------------------------------------
# MOSTRAR Y GUARDAR RESULTADOS
# ---------------------------------------------------------
print("\n=====================================")
print("ESTADÍSTICA DESCRIPTIVA — PDI")
print("=====================================")
print(summary_pdi)

summary_pdi.to_excel(
    "tabla_estadistica_PDI_SNEDDS.xlsx",
    index=False
)

# ==========================================
# FIGURA PROFESIONAL: TAMAÑO + PDI (CORREGIDA)
# ==========================================

# Orden fisiológico
orden_fases = ["Inicial", "Boca", "Estómago", "Intestino"]

# Asegurar orden correcto en ambas tablas
summary["fase_fisiologica"] = pd.Categorical(
    summary["fase_fisiologica"],
    categories=orden_fases,
    ordered=True
)
summary = summary.sort_values("fase_fisiologica")

summary_pdi["fase_fisiologica"] = pd.Categorical(
    summary_pdi["fase_fisiologica"],
    categories=orden_fases,
    ordered=True
)
summary_pdi = summary_pdi.sort_values("fase_fisiologica")

# Crear figura con dos paneles
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ---------------------------
# Panel 1: Tamaño de partícula
# ---------------------------
axes[0].errorbar(
    summary["fase_fisiologica"],
    summary["media_tamano"],
    yerr=summary["std_tamano"],
    marker='o',
    capsize=5,
    linewidth=2
)

axes[0].set_ylabel("Tamaño de partícula (nm)")
axes[0].set_title("Evolución estructural del SNEDDS durante la digestión in vitro")
axes[0].grid(True, alpha=0.3)

# ---------------------------
# Panel 2: PDI
# ---------------------------
axes[1].errorbar(
    summary_pdi["fase_fisiologica"],
    summary_pdi["media_pdi"],
    yerr=summary_pdi["std_pdi"],
    marker='s',
    capsize=5,
    linewidth=2,
    color="tab:orange"
)

axes[1].set_ylabel("Índice de polidispersidad (PDI)")
axes[1].set_xlabel("Fase fisiológica")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
 
# ==========================================
# FIGURA PROFESIONAL FINAL
# ==========================================

orden_fases = ["Inicial", "Boca", "Estómago", "Intestino"]

# Ordenar tamaño
summary["fase_fisiologica"] = pd.Categorical(
    summary["fase_fisiologica"],
    categories=orden_fases,
    ordered=True
)
summary = summary.sort_values("fase_fisiologica")

# Ordenar PDI
summary_pdi["fase_fisiologica"] = pd.Categorical(
    summary_pdi["fase_fisiologica"],
    categories=orden_fases,
    ordered=True
)
summary_pdi = summary_pdi.sort_values("fase_fisiologica")

# Crear figura
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Tamaño
axes[0].errorbar(
    summary["fase_fisiologica"],
    summary["media_tamano"],
    yerr=summary["std_tamano"],
    marker='o',
    capsize=5,
    linewidth=2
)

axes[0].set_ylabel("Tamaño de partícula (nm)")
axes[0].set_title("Evolución estructural del SNEDDS durante la digestión in vitro")
axes[0].grid(True, alpha=0.3)

# PDI
axes[1].errorbar(
    summary_pdi["fase_fisiologica"],
    summary_pdi["media_pdi"],
    yerr=summary_pdi["std_pdi"],
    marker='s',
    capsize=5,
    linewidth=2,
    color="tab:orange"
)

axes[1].set_ylabel("Índice de polidispersidad (PDI)")
axes[1].set_xlabel("Fase fisiológica")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
