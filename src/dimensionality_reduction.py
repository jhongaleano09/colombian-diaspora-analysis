import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

def apply_pca(data, n_components):
    """
    Aplica PCA a los datos con un número específico de componentes.

    Args:
        data (pd.DataFrame or np.ndarray): Los datos escalados a transformar.
        n_components (int): El número de componentes principales a retener.

    Returns:
        pd.DataFrame: Un DataFrame con los datos transformados por PCA.
    """
    print(f"Aplicando PCA para reducir a {n_components} componentes...")
    
    pca = PCA(n_components=n_components)
    
    try:
        data_transformed = pca.fit_transform(data)
        
        # Crear un DataFrame con nombres de columna descriptivos
        pca_column_names = [f'PC_{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(data=data_transformed, columns=pca_column_names)
        
        print("Transformación PCA finalizada exitosamente.")
        return df_pca

    except MemoryError:
        print("\nERROR DE MEMORIA: El dataset es muy grande para PCA estándar.")
        print("Considera usar IncrementalPCA o una máquina con más RAM.")
        # Aquí podrías implementar una lógica de fallback a IncrementalPCA si lo deseas.
        raise
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la transformación PCA: {e}")
        raise