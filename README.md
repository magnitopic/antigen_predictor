# antigen_predictor

Clasificador de antigenicidad de proteínas usando Machine Learning.  
Proyecto educativo desarrollado con Python, scikit-learn, Biopython y desplegado en Streamlit.

**App desplegada:** [https://antigenpredictor-apa.streamlit.app](https://antigenpredictor-apa.streamlit.app)

---

Dado un conjunto de proteínas de un patógeno, el modelo predice cuáles tienen más probabilidad de ser antigénicas (reconocidas por el sistema inmune humano), usando características calculadas a partir de su secuencia de aminoácidos.

---

## ¿Qué hace este proyecto?

Dado un archivo FASTA con secuencias de proteínas de un patógeno, el sistema predice cuáles tienen mayor probabilidad de ser antigénicas, es decir, de ser reconocidas por el sistema inmune humano.

La salida es una tabla con un score de antigenicidad (entre 0 y 1) por proteína y un gráfico de barras ordenado de mayor a menor score, con opción de descarga en CSV.

Este tipo de herramienta puede usarse como filtro rápido para priorizar candidatos vacunales in silico, reduciendo el tiempo de búsqueda inicial de antígenos. No reemplaza ensayos experimentales ni ensayos clínicos.

---

## Contexto y motivación

El proyecto es un ejercicio educativo de Machine Learning aplicado a bioinformática. El objetivo es construir un pipeline completo: desde la obtención y limpieza de datos reales hasta el despliegue de una interfaz web funcional.

Los patógenos usados como caso de estudio son **SARS-CoV-2** e **Influenza A**, por su relevancia clínica y por la abundancia de datos experimentales disponibles en bases de datos públicas.

---

## Dataset

**Fuente:** [IEDB — Immune Epitope Database](https://www.iedb.org)

IEDB es una base de datos pública y gratuita mantenida por el National Institute of Allergy and Infectious Diseases (NIAID). Contiene datos experimentales de epítopos: fragmentos de proteínas reconocidos por el sistema inmune en ensayos de laboratorio.

**Archivos utilizados:**

| Archivo IEDB | Contenido | Tamaño descomprimido |
|---|---|---|
| `tcell_full_v3.csv` | Ensayos de respuesta de células T | ~1.3 GB |
| `bcell_full_v3.csv` | Ensayos de respuesta de células B | ~2.6 GB |

> **Nota:** Estos archivos no están incluidos en el repositorio por su tamaño. Deben descargarse desde [iedb.org/database_export_v3.php](https://www.iedb.org/database_export_v3.php). El archivo filtrado resultante (`data/raw/iedb_sars_flu_filtered.csv`, 29 MB) sí está incluido.

**Estrategia de construcción del dataset:**

1. Descarga de `tcell_full_v3.csv` y `bcell_full_v3.csv` desde IEDB.
2. Filtrado local por organismo (SARS-CoV-2 e Influenza A), extrayendo solo las columnas necesarias. Resultado: `data/raw/iedb_sars_flu_filtered.csv` (158,289 filas, 29 MB).
3. Agrupación por proteína: una proteína es `label = 1` si tiene al menos un ensayo positivo, `label = 0` si solo tiene ensayos negativos.
4. Descarga de secuencias FASTA desde la API de NCBI Entrez para cada proteína.
5. Cálculo de features fisicoquímicas con Biopython.
6. Resultado: `data/processed/dataset.csv` con 1,310 proteínas.

**Distribución del dataset final:**

| Label | Proteínas | Descripción |
|---|---|---|
| 1 (antigénica) | 1,149 | Al menos un epítopo positivo en IEDB |
| 0 (no antigénica) | 161 | Solo epítopos negativos en IEDB |

---

## Features del modelo

Calculadas con **Biopython** a partir de la secuencia de aminoácidos de cada proteína:

| Feature | Descripción |
|---|---|
| Longitud | Número de aminoácidos |
| Peso molecular | En Daltons |
| Punto isoeléctrico | pH al que la carga neta de la proteína es cero |
| Hidrofobicidad media | Índice GRAVY |
| Composición de aminoácidos | Porcentaje de cada uno de los 20 aminoácidos estándar |

Total: **24 features** por proteína (4 fisicoquímicas + 20 de composición).

No se usan embeddings de modelos de lenguaje de proteínas ni features de estructura 3D.

---

## Modelo

- **Algoritmo principal:** Random Forest (200 árboles, scikit-learn)
- **Validación:** Cross-validation estratificada (k=5)
- **Desbalance de clases:** gestionado con `class_weight='balanced'`
- **Métricas:** AUC-ROC y F1-score
- **Baselines:** clasificador de mayoría y Logistic Regression
- **Guardado:** `models/model.pkl` con joblib

**Resultados obtenidos:**

| Modelo | AUC-ROC | F1 |
|---|---|---|
| Mayoría (baseline) | 0.500 ± 0.000 | — |
| Logistic Regression | 0.626 ± 0.043 | 0.795 ± 0.026 |
| **Random Forest** | **0.719 ± 0.050** | **0.935 ± 0.004** |

---

## Estructura del repositorio

```
antigen_predictor/
│
├── app/
│   └── app.py                          ← interfaz web Streamlit
│
├── data/
│   ├── raw/
│   │   └── iedb_sars_flu_filtered.csv  ← ensayos filtrados de IEDB (29 MB)
│   └── processed/
│       ├── protein_labels.csv          ← una fila por proteína con label
│       └── dataset.csv                 ← features + label, listo para entrenar
│
├── models/
│   ├── model.pkl                       ← modelo entrenado
│   ├── roc_curves.png                  ← curvas ROC por fold
│   └── feature_importance.png          ← importancia de features
│
├── notebooks/
│   ├── 00_data_acquisition.ipynb       ← documentación del preprocesamiento
│   ├── 01_dataset_exploration.ipynb    ← exploración del dataset filtrado
│   ├── 02_dataset_construction.ipynb   ← descarga de secuencias y features
│   ├── 03_features_and_model.ipynb     ← entrenamiento y evaluación
│   └── 04_app.ipynb                    ← despliegue de la app
│
├── glossary_of_terms.md
├── project_viability_analysis.md
├── requirements.txt
└── README.md
```

---

## Requisitos

```
Python 3.9+
biopython
scikit-learn
pandas
numpy
matplotlib
streamlit
joblib
```

Instalación:

```bash
pip install -r requirements.txt
```

---

## Cómo usar

**Ejecutar los notebooks** (en orden, desde JupyterLab):

```bash
jupyter lab
```

El NB00 es solo documentación. Los notebooks ejecutables son el 01, 02, 03 y 04.

**Lanzar la app localmente:**

```bash
cd app
streamlit run app.py
```

**App pública (Streamlit Community Cloud):**  
[https://antigenpredictor-apa.streamlit.app](https://antigenpredictor-apa.streamlit.app)

---

## Limitaciones

- El modelo está entrenado únicamente con datos de SARS-CoV-2 e Influenza A. Su capacidad de generalización a otros patógenos es desconocida.
- Las features usadas son exclusivamente de secuencia. No se considera estructura 3D, glicosilación ni procesamiento celular.
- El dataset está desbalanceado (7:1), gestionado con `class_weight='balanced'` pero con solo 161 ejemplos negativos.
- Un score alto no garantiza que la proteína sea un buen antígeno vacunal. Es un filtro orientativo, no un predictor clínico.

---

## Equipo

Proyecto desarrollado por tres estudiantes como ejercicio educativo de Machine Learning.

---

## Licencia

MIT
