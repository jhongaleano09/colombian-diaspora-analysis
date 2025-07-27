from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def run_kmeans(data, n_clusters):
    """Ejecuta el algoritmo K-Means y devuelve las etiquetas y el score de silueta."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data)
    score = silhouette_score(data, labels)
    print(f"K-Means con {n_clusters} clusters. Score de Silueta: {score:.3f}")
    return labels, score

# Aquí podrías añadir funciones para encontrar el K óptimo (método del codo)
# o para ejecutar y evaluar DBSCAN.