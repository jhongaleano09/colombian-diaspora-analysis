# Análisis de la Diáspora Colombiana con Machine Learning

## Resumen Ejecutivo

Este proyecto utiliza un dataset público del Ministerio de Relaciones Exteriores de Colombia, con **1.65 millones de registros**, para descubrir patrones y perfiles ocultos dentro de la población migrante colombiana. Mediante el uso de algoritmos de **clustering no supervisado (K-Means, DBSCAN)** y modelos **supervisados (Random Forest)**, este análisis transforma datos administrativos en insights accionables sobre las dinámicas geográficas, demográficas y profesionales de la diáspora.

El objetivo principal es demostrar cómo la ciencia de datos puede revelar segmentaciones complejas que los análisis descriptivos tradicionales no capturan, sentando un precedente para futuros estudios migratorios basados en datos.

## Problema y Objetivo

Las estadísticas tradicionales sobre migración ofrecen una visión general, pero fallan en capturar la **heterogeneidad** de la población. ¿Existen perfiles migratorios distintivos definidos por combinaciones de edad, nivel educativo y destino?

**Objetivo General:** Aplicar técnicas de Machine Learning para segmentar la diáspora colombiana, identificar perfiles migratorios latentes y evaluar comparativamente los enfoques no supervisados y supervisados para generar un entendimiento más profundo del fenómeno.

## Dataset

El proyecto utiliza el dataset **"Colombianos registrados en el exterior"**, que es público y está disponible en el portal de Datos Abiertos de Colombia. Esto garantiza la total reproducibilidad del análisis.

- **Fuente:** [Portal de Datos Abiertos de Colombia](https://www.datos.gov.co/Estad-sticas-Nacionales/Colombianos-registrados-en-el-exterior/y399-rzwf/about_data)
- **Tamaño:** 1.65 millones de registros y 17 variables.
- **Licencia:** Abierta.

## Stack Tecnológico

- **Lenguaje:** Python
- **Librerías Principales:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, JupyterLab

## Estructura del Repositorio

El proyecto está organizado siguiendo las mejores prácticas para proyectos de ciencia de datos, asegurando modularidad y reproducibilidad:

```
├── data/              # Almacenamiento de datasets (excluidos por .gitignore)
├── notebooks/         # Notebooks de exploración y prototipado (EDA)
├── reports/figures/   # Visualizaciones y gráficos generados
├── src/               # Código fuente modularizado
│   ├── data_preprocessing.py
│   ├── clustering_analysis.py
│   ├── supervised_modeling.py
│   └── visualization.py
├── .gitignore         # Archivos a ignorar por Git
└── requirements.txt   # Dependencias del proyecto
```

## Cómo Ejecutar el Proyecto

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/jhongaleano09/colombian-diaspora-analysis.git](https://github.com/jhongaleano09/colombian-diaspora-analysis.git)
    cd colombian-diaspora-analysis
    ```

2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Descargar el dataset:**
    Visita el [Portal de Datos Abiertos de Colombia](https://www.datos.gov.co/Estad-sticas-Nacionales/Colombianos-registrados-en-el-exterior/y399-rzwf/about_data), descarga el archivo CSV y guárdalo en el directorio `data/raw/`.

5.  **Ejecutar el análisis:**
    El flujo principal del proyecto puede ser ejecutado desde el notebook `notebooks/1.0-EDA_and_Prototyping.ipynb`, el cual ahora importa y utiliza las funciones modulares del directorio `src/`.

## Metodología

1.  **Análisis Exploratorio de Datos (EDA):** Se realizó un análisis inicial para comprender las distribuciones, identificar valores atípicos y visualizar las principales tendencias demográficas y geográficas.
2.  **Preprocesamiento de Datos:** El código en `src/data_preprocessing.py` se encarga de la limpieza, manejo de valores nulos/no estándar, y la transformación de variables para prepararlas para el modelado.
3.  **Análisis de Clustering (No Supervisado):** Se aplicaron algoritmos como K-Means para segmentar a la población en clusters con características similares. El objetivo es descubrir "perfiles migratorios" basados en datos.
4.  **Modelado Supervisado (Clasificación):** Se entrenó un modelo de Random Forest para predecir el `Nivel Académico` de un migrante basado en sus otras características, validando los hallazgos del clustering y explorando la predictibilidad de las variables.

## Hallazgos Clave

*(Esta sección es crucial. Debes actualizarla con los resultados más impactantes de tu análisis. Por ejemplo:)*

-   **Perfil 1: "Jóvenes Profesionales en Europa"**: Se identificó un cluster significativo de individuos entre 25-35 años, con niveles de posgrado, concentrados en España y Alemania.
-   **Visualización Destacada**: El siguiente mapa muestra la concentración geográfica de los clusters identificados...

    *(Inserta aquí una de tus mejores visualizaciones guardada en `reports/figures/`)*
    
    `![Concentración de Clusters](reports/figures/mapa_clusters.png)`

## Posibles Mejoras

-   Implementar algoritmos de clustering más avanzados como HDBSCAN o GMMs.
-   Enriquecer el dataset con datos socioeconómicos de los países de destino (ej. PIB, tasa de desempleo).
-   Desplegar el modelo predictivo como una API simple usando Flask o FastAPI.
-   Incorporar herramientas de MLOps como MLflow para el seguimiento de experimentos.