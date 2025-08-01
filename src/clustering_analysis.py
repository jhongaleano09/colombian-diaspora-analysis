from sklearn.cluster import KMeans, DBSCAN
import pandas as pd
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def run_kmeans(data, n_clusters):
    """Ejecuta el algoritmo K-Means y devuelve las etiquetas y el score de silueta."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    print(f"K-Means con {n_clusters} clusters. Score de Silueta: {score:.3f}")
    return labels, score

# clustering_analysis.py
# Este archivo contiene funciones para realizar análisis de clustering, incluyendo K-Means y MiniBatchKMeans.

def find_optimal_k_minibatch(data, k_range, batch_size=4096, sample_size_metrics=100000, random_state=42):
    """
    Evalúa diferentes valores de k para MiniBatchKMeans y devuelve las métricas.
    Utiliza un submuestreo para métricas computacionalmente costosas.

    Args:
        data (pd.DataFrame): Datos para el clustering (ej. componentes principales).
        k_range (range): Rango de valores k a probar.
        batch_size (int): Tamaño del lote para MiniBatchKMeans.
        sample_size_metrics (int): Tamaño de la muestra para calcular Silhouette/DB/CH.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        pd.DataFrame: Un DataFrame con las métricas para cada valor de k.
    """
    # Preparar la muestra para métricas
    if data.shape[0] > sample_size_metrics:
        print(f"Calculando métricas (Silhouette/DB/CH) sobre una muestra de {sample_size_metrics} puntos.")
        data_sample = data.sample(n=sample_size_metrics, random_state=random_state)
    else:
        print("Calculando métricas (Silhouette/DB/CH) sobre el dataset completo.")
        data_sample = data

    metrics = []
    print(f"\nEvaluando k en el rango {list(k_range)}...")
    start_time = time.time()

    for k in k_range:
        print(f"  Probando k={k}...")
        mbk = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size,
            random_state=random_state,
            n_init='auto',
            max_iter=300
        )
        
        # Ajustar al dataset completo para obtener inercia y un modelo robusto
        mbk.fit(data)
        
        # Predecir sobre la muestra para calcular métricas de forma eficiente
        sample_labels = mbk.predict(data_sample)
        
        # Calcular métricas
        if len(np.unique(sample_labels)) > 1:
            sil_score = silhouette_score(data_sample, sample_labels)
            db_score = davies_bouldin_score(data_sample, sample_labels)
            ch_score = calinski_harabasz_score(data_sample, sample_labels)
        else:
            sil_score, db_score, ch_score = np.nan, np.nan, np.nan

        metrics.append({
            'k': k,
            'inertia': mbk.inertia_,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        })

    end_time = time.time()
    print(f"... Evaluación de k completada en {end_time - start_time:.2f} segundos.")
    
    return pd.DataFrame(metrics)

def run_minibatch_kmeans(data, n_clusters, batch_size=4096, random_state=42):
    """
    Ejecuta MiniBatchKMeans con un número específico de clusters.

    Args:
        data (pd.DataFrame): Los datos a agrupar.
        n_clusters (int): El número de clusters (k).
        batch_size (int): Tamaño del lote para MiniBatchKMeans.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        tuple: Una tupla conteniendo (cluster_labels, cluster_centroids).
    """
    print(f"\nAplicando MiniBatchKMeans con k={n_clusters} al dataset completo...")
    start_time = time.time()

    final_mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init='auto',
        max_iter=300
    )
    
    labels = final_mbk.fit_predict(data)
    centroids = final_mbk.cluster_centers_

    end_time = time.time()
    print(f"... Clustering final completado en {end_time - start_time:.2f} segundos.")
    
    return labels, centroids


# src/clustering_analysis.py
# Añade de forma segura las etiquetas de cluster a un DataFrame original.

def get_cluster_summary_stats(df, cluster_col, numeric_cols):
    """
    Calcula estadísticas descriptivas (media y std) para columnas numéricas,
    agrupadas por cluster.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos y la columna de cluster.
        cluster_col (str): Nombre de la columna de cluster.
        numeric_cols (list): Lista de nombres de columnas numéricas para analizar.

    Returns:
        pd.DataFrame: Un DataFrame con las estadísticas de media y desviación estándar
                      para cada cluster.
    """
    print(f"\nCalculando estadísticas descriptivas para las columnas: {numeric_cols}")
    
    if cluster_col not in df.columns:
        raise KeyError(f"La columna de cluster '{cluster_col}' no se encuentra en el DataFrame.")
        
    # Verificar que todas las columnas numéricas existan
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Las siguientes columnas no se encontraron en el DataFrame: {missing_cols}")

    cluster_stats = df.groupby(cluster_col)[numeric_cols].agg(['mean', 'std'])
    
    return cluster_stats


# En src/clustering_analysis.py

def display_categorical_distribution_by_cluster(df, target_col, cluster_col='Cluster', top_n=None):
    """
    Calcula y muestra la distribución porcentual de una variable categórica
    dentro de cada cluster.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): Nombre de la columna categórica a analizar.
        cluster_col (str): Nombre de la columna de cluster (default: 'Cluster').
        top_n (int, optional): Si se especifica, muestra solo las 'top_n'
                               categorías más frecuentes. Por defecto es None (mostrar todas).
    """
    print(f"--- Distribución de '{target_col}' por Cluster (%) ---")

    if cluster_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"Una o ambas columnas ('{cluster_col}', '{target_col}') no se encuentran en el DataFrame.")

    # Calcular la tabla de contingencia normalizada por filas (índice)
    distribution_table = pd.crosstab(df[cluster_col], df[target_col], normalize='index') * 100

    # Si se especifica top_n, filtrar por las categorías más comunes del dataset
    if top_n and top_n > 0:
        top_categories = df[target_col].value_counts().nlargest(top_n).index
        available_categories = [cat for cat in top_categories if cat in distribution_table.columns]
        distribution_table = distribution_table[available_categories]
        print(f"(Mostrando las {len(available_categories)} categorías más frecuentes)")

    display(distribution_table.round(1))