import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
