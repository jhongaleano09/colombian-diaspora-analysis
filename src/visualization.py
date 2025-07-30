import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import os # Para asegurar que el directorio de figuras exista


def plot_top_countries(df, top_n=15):
    """Grafica el top N de países con más colombianos registrados."""
    plt.figure(figsize=(12, 8))
    country_counts = df['País'].value_counts().head(top_n)
    sns.barplot(x=country_counts.values, y=country_counts.index, palette='viridis')
    plt.title(f'Top {top_n} Países de Residencia de Colombianos en el Exterior')
    plt.xlabel('Cantidad de Registros')
    plt.ylabel('País')
    plt.tight_layout()
    # Guardar la figura es una buena práctica
    plt.savefig('reports/figures/top_countries_distribution.png')
    plt.show()

# src/visualization.py

def display_descriptive_stats(df):
    """Muestra estadísticas descriptivas para variables numéricas y categóricas."""
    print("--- Estadísticas Descriptivas Numéricas ---")
    print(df.describe())
    print("\n" + "="*50 + "\n")
    
    print("--- Estadísticas Descriptivas Categóricas (baja cardinalidad) ---")
    cols_categoricas = df.select_dtypes(include=['object', 'category']).columns
    cols_alta_cardinalidad = ['Ciudad de Residencia', 'Ciudad de Nacimiento', 'Sub Area Conocimiento', 'Código ISO país'] 
    cols_a_describir = [col for col in cols_categoricas if col not in cols_alta_cardinalidad]
    print(df[cols_a_describir].describe(include='all'))

# src/visualization.py

def plot_weighted_distribution(df, column, top_n=None, title=None, xlabel=None, ylabel=None, exclude_list=None):
    """     Genera un gráfico de barras ponderado por 'Cantidad de personas'.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        column (str): Nombre de la columna a agrupar.
        top_n (int, optional): Número de categorías a mostrar. Si es None, muestra todas. Defaults to None.
        title (str, optional): Título del gráfico. Defaults to None.
        xlabel (str, optional): Etiqueta del eje X. Defaults to None.
        ylabel (str, optional): Etiqueta del eje Y. Defaults to None.
        exclude_list (list, optional): Lista de valores a excluir de la columna. Defaults to None.
    """
    # --- Preparación de Datos ---
    # Usar una copia para evitar modificar el DataFrame original
    df_plot = df.copy()

    # Excluir valores si se proporciona una lista
    if exclude_list:
        df_plot = df_plot[~df_plot[column].isin(exclude_list)]

    if df_plot.empty:
        print(f"No hay datos para graficar en '{column}' después de filtrar.")
        return

    # Agrupar por la columna y sumar la 'Cantidad de personas'
    data = df_plot.groupby(column)['Cantidad de personas'].sum().sort_values(ascending=False)

    # Seleccionar el top N si se especifica
    if top_n:
        data = data.head(top_n)

    # --- Creación del Gráfico ---
    plt.figure(figsize=(14, 7))
    data.plot(kind='bar', color=sns.color_palette("viridis", len(data)))
    
    # --- Títulos y Etiquetas (con valores por defecto) ---
    plt.title(title or f'Distribución por {column}', fontsize=16)
    plt.xlabel(xlabel or column, fontsize=12)
    plt.ylabel(ylabel or 'Número Estimado de Personas', fontsize=12)
    
    plt.xticks(rotation=70, ha='right')
    plt.tight_layout() # Ajusta el gráfico para que no se corten las etiquetas
    plt.show()


def plot_weighted_age_histogram(df):
    """
    Genera un histograma de la edad ponderado por 'Cantidad de personas'.
    """
    plt.figure(figsize=(12, 6))
    
    # Calcular el promedio ponderado
    weighted_avg_edad = np.average(df['Edad (años)'], weights=df['Cantidad de personas'])
    
    # Crear el histograma ponderado
    sns.histplot(data=df, x='Edad (años)', weights='Cantidad de personas', kde=True, bins=30)
    
    # Añadir línea de promedio ponderado
    plt.axvline(weighted_avg_edad, color='red', linestyle='dashed', linewidth=1.5, label=f'Promedio Ponderado: {weighted_avg_edad:.1f} años')
    
    plt.title('Distribución Ponderada de Edad de Migrantes Colombianos', fontsize=16)
    plt.xlabel('Edad (años)', fontsize=12)
    plt.ylabel('Número Estimado de Personas', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

# src/visualization.py
import os

def plot_correlation_heatmap(df, title='Matriz de Correlación de Variables Numéricas', save_path=None):
    """
    Calcula y grafica una matriz de correlación (heatmap) para las 
    columnas numéricas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        title (str, optional): Título del gráfico.
        save_path (str, optional): Ruta para guardar la imagen del gráfico. 
                                   Si es None, el gráfico solo se muestra.
    """
    print("Generando heatmap de correlación...")
    
    # 1. Seleccionar solo columnas numéricas para el análisis
    df_numeric = df.select_dtypes(include=np.number)
    
    # 2. Calcular la matriz de correlación
    correlation_matrix = df_numeric.corr()
    
    # 3. Configurar el tamaño y crear el heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, 
        annot=True,      # Mostrar los valores de correlación en las celdas
        cmap='coolwarm', # Esquema de color divergente
        fmt=".2f",       # Formato de dos decimales
        linewidths=.5
    )
    
    # 4. Añadir título y ajustar el layout
    plt.title(title, fontsize=16)
    plt.tight_layout()  # Ajusta el gráfico para evitar que las etiquetas se superpongan
    
    # 5. Guardar la figura si se especifica una ruta
    if save_path:
        # Asegurarse de que el directorio de destino exista
        output_dir = os.path.dirname(save_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(save_path, dpi=300) # Guardar con buena resolución
        print(f"Gráfico guardado en: {save_path}")
        
    # 6. Mostrar el gráfico
    plt.show()



# src/visualization.py
# Análisis de varianza explicada por PCA

def plot_pca_variance_analysis(data):
    """
    Realiza un análisis de PCA para visualizar la varianza explicada acumulada y
    ayudar a determinar el número óptimo de componentes.

    Args:
        data (pd.DataFrame or np.ndarray): El conjunto de datos escalado.
    """
    print("Iniciando análisis de PCA para determinar la varianza explicada...")
    
    try:
        # Usamos PCA sin n_components para calcular la varianza de todos los componentes
        pca_explorer = PCA()
        pca_explorer.fit(data)

        # Calcular la varianza explicada acumulada
        cumulative_explained_variance = np.cumsum(pca_explorer.explained_variance_ratio_)

        # --- Visualización ---
        plt.figure(figsize=(12, 7))
        plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, 
                 marker='.', linestyle='-', color='b', markersize=4)
        plt.title('Varianza Explicada Acumulada por Componentes Principales', fontsize=16)
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Varianza Explicada Acumulada')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Líneas de referencia para umbrales comunes
        thresholds = {0.90: 'r', 0.95: 'g', 0.99: 'purple'}
        for thresh, color in thresholds.items():
            n_components = np.argmax(cumulative_explained_variance >= thresh) + 1
            plt.axhline(y=thresh, color=color, linestyle='--', label=f'{int(thresh*100)}% Varianza ({n_components} comps)')
            print(f"Componentes para explicar al menos el {int(thresh*100)}% de la varianza: {n_components}")

        plt.legend(loc='best')
        plt.ylim(top=1.05)
        
        # Guardar la figura
        plt.savefig('reports/figures/pca_explained_variance.png')
        plt.show()

    except MemoryError:
        print("\nERROR DE MEMORIA durante el análisis de varianza.")
        print("Sugerencia: Ejecuta esta función sobre una muestra aleatoria del dataset para estimar los componentes.")
        print("Ejemplo: `plot_pca_variance_analysis(data.sample(n=100000))`")
        raise
    except Exception as e:
        print(f"Ocurrió un error inesperado durante el análisis de PCA: {e}")
        raise


# src/visualization.py


def plot_k_selection_metrics(metrics_df, save_path='reports/figures/k_selection_metrics.png'):
    """
    Grafica las métricas de evaluación de k (Inertia, Silhouette, etc.).

    Args:
        metrics_df (pd.DataFrame): DataFrame con columnas 'k', 'inertia', 'silhouette', etc.
        save_path (str): Ruta para guardar la figura.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True) # Crea el directorio si no existe
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    k_range = metrics_df['k']

    # 1. Gráfico del Codo (Inertia)
    axes[0].plot(k_range, metrics_df['inertia'], marker='o', linestyle='-')
    axes[0].set_title('Método del Codo (Inertia)', fontsize=14)
    axes[0].set_xlabel('Número de Clusters (k)')
    axes[0].set_ylabel('Inertia (WCSS)')
    axes[0].grid(True)
    axes[0].set_xticks(k_range)

    # 2. Gráfico de Silhouette Score
    axes[1].plot(k_range, metrics_df['silhouette'], marker='o', linestyle='-')
    axes[1].set_title('Silhouette Score Promedio', fontsize=14)
    axes[1].set_xlabel('Número de Clusters (k)')
    axes[1].set_ylabel('Silhouette Score (mayor es mejor)')
    axes[1].grid(True)
    axes[1].set_xticks(k_range)

    # 3. Gráfico de Davies-Bouldin Index
    axes[2].plot(k_range, metrics_df['davies_bouldin'], marker='o', linestyle='-')
    axes[2].set_title('Davies-Bouldin Index', fontsize=14)
    axes[2].set_xlabel('Número de Clusters (k)')
    axes[2].set_ylabel('DB Score (menor es mejor)')
    axes[2].grid(True)
    axes[2].set_xticks(k_range)

    plt.suptitle('Métricas de Evaluación para Selección de k', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_cluster_results(df, x_col, y_col, hue_col, centroids=None, save_path='reports/figures/cluster_visualization.png'):
    """
    Visualiza los resultados del clustering en un scatter plot 2D.

    Args:
        df (pd.DataFrame): DataFrame con los datos y las etiquetas del cluster.
        x_col (str): Nombre de la columna para el eje X.
        y_col (str): Nombre de la columna para el eje Y.
        hue_col (str): Nombre de la columna para el color (etiquetas del cluster).
        centroids (pd.DataFrame, optional): DataFrame de los centroides a graficar.
        save_path (str): Ruta para guardar la figura.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        x=df[x_col], 
        y=df[y_col], 
        hue=df[hue_col],
        palette='viridis', 
        alpha=0.5, 
        s=10, # 's' pequeño para datasets grandes
        legend='full'
    )
    
    if centroids is not None:
        plt.scatter(
            centroids[x_col], 
            centroids[y_col],
            marker='X', 
            s=200, 
            c='red', 
            edgecolor='black', 
            label='Centroides'
        )

    plt.title(f'Visualización de Clusters en {x_col} vs {y_col}', fontsize=16)
    plt.xlabel(f'Componente Principal ({x_col})')
    plt.ylabel(f'Componente Principal ({y_col})')
    plt.legend(title='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()