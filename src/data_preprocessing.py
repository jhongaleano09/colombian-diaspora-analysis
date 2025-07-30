import pandas as pd
import numpy as np

def load_data(file_path):
    """Carga el dataset desde una ruta de archivo CSV."""
    print(f"Cargando dataset desde: {file_path}")
    df = pd.read_csv(file_path)
    print("¡Dataset cargado exitosamente!")
    return df

# src/data_preprocessing.py

def clean_data(df):
    """
    Realiza la limpieza inicial de datos: convierte fechas, maneja valores
    no estándar y elimina registros inconsistentes.
    """
    print("Iniciando limpieza de datos...")
    
    # Reemplazar valores no estándar (-1) con NaN para un manejo adecuado
    df.replace(-1, np.nan, inplace=True)
    
    # Imputar 'Edad (años)' con la media
    mean_age = df['Edad (años)'].mean()
    df['Edad (años)'].fillna(mean_age, inplace=True)
    
    # Imputar 'Estatura (CM)' con un valor constante (ej. media o mediana)
    # Aquí se usa la media como ejemplo.
    mean_height = df['Estatura (CM)'].mean()
    df['Estatura (CM)'].fillna(mean_height, inplace=True)
    
    # Eliminar filas con valores nulos en columnas críticas si es necesario
    # (En este caso, no se eliminan filas para mantener el tamaño del dataset)

    # Convertir 'Fecha de Registro' a datetime y crear columna de año
    df['Fecha de Registro'] = pd.to_datetime(df['Fecha de Registro'], format='%Y-%m', errors='coerce')
    df.dropna(subset=['Fecha de Registro'], inplace=True) # Eliminar fechas no válidas
    df['Año Registro'] = df['Fecha de Registro'].dt.year
    
    # Filtrar registros inconsistentes (ej. 'Cantidad de personas' <= 0)
    df = df[df['Cantidad de personas'] > 0].copy()
    
    print(f"Limpieza finalizada. Forma del DataFrame: {df.shape}")
    return df

# src/data_preprocessing.py
# Elimina columnas innecesarias del DataFrame

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina un conjunto predefinido de columnas redundantes o no deseadas 
    del DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.

    Returns:
        pd.DataFrame: El DataFrame sin las columnas especificadas.
    """
    columns_to_drop = [
        'Código ISO país',
        'Edad (años)',
        'Fecha de Registro',
        'Día Registro',
        'Localización',
        'Ciudad de Residencia',
        'Estatura (CM)'
    ]
    
    # Filtra solo las columnas que realmente existen en el DataFrame para evitar errores
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    
    if existing_columns_to_drop:
        print(f"--- Columnas eliminadas: {', '.join(existing_columns_to_drop)} ---")
        return df.drop(columns=existing_columns_to_drop)
    else:
        print("--- Ninguna de las columnas especificadas para eliminar fue encontrada ---")
        return df.copy()


# src/data_preprocessing.py
# Procesa la columna de lugar de nacimiento para extraer el departamento o país


def clean_birthplace_data(df: pd.DataFrame, column_name: str = 'Ciudad de Nacimiento') -> pd.DataFrame:
    """
    Procesa una columna de lugar de nacimiento para extraer el departamento (si es de Colombia)
    o el país. Crea una nueva columna 'Departamento_o_pais_de_nacimiento' y elimina la original.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        column_name (str): El nombre de la columna que contiene los datos de nacimiento.

    Returns:
        pd.DataFrame: El DataFrame con la columna transformada.
    """
    
    # 1. Copiamos el dataframe para evitar advertencias de SettingWithCopyWarning
    df_copy = df.copy()

    # 2. Se define la función auxiliar (helper) dentro de la función principal
    def _procesar_lugar_nacimiento(texto_lugar):
        """
        Procesa el string de lugar de nacimiento con lógica diferenciada.
        - Si empieza con 'COLOMBIA' y tiene formato 'COLOMBIA/DEPARTAMENTO/CIUDAD', devuelve 'DEPARTAMENTO'.
        - Si empieza con otro país (y es un string válido), devuelve el 'PAIS'.
        - En otros casos (NaN, formato inválido, etc.), devuelve 'Otros'.
        """
        if not isinstance(texto_lugar, str) or not texto_lugar.strip():
            return 'Otros'

        partes = [p.strip() for p in texto_lugar.split('/') if p.strip()]

        if not partes:
            return 'Otros'

        pais = partes[0]

        if pais.upper() == 'COLOMBIA':
            return partes[1] if len(partes) == 3 else 'Otros'
        else:
            return pais

    # 3. Aplicar la función y crear la nueva columna
    new_col_name = 'Departamento_o_pais_de_nacimiento'
    df_copy[new_col_name] = df_copy[column_name].apply(_procesar_lugar_nacimiento)

    # 4. Eliminar la columna original
    df_copy = df_copy.drop(columns=[column_name])
    
    print(f"Columna '{column_name}' procesada. Nueva columna '{new_col_name}' creada y original eliminada.")
    
    return df_copy



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def select_features(df, cols_to_drop):
    """
    Selecciona las características para el modelado y las clasifica en numéricas y categóricas.
    """
    df_model = df.drop(columns=cols_to_drop)
    numerical_features = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df_model.select_dtypes(include=['object']).columns.tolist()
    print("Características seleccionadas para el modelado.")
    return df_model, numerical_features, categorical_features

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """
    Crea un pipeline de ColumnTransformer para escalar datos numéricos y codificar categóricos.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            # CORRECCIÓN: sparse_output=True para generar una matriz dispersa eficiente.
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)
        ],
        remainder='passthrough'
    )
    print("Pipeline de preprocesamiento creado.")
    return preprocessor


# src/data_preprocessing.py
# Ejecuta MiniBatchKMeans para encontrar el número óptimo de clusters

def add_cluster_labels(original_df, pca_df_with_clusters, cluster_col_name, new_col_name='Cluster'):
    """
    Añade de forma segura las etiquetas de cluster a un DataFrame original.
    
    Asegura la alineación correcta reseteando los índices de ambos DataFrames
    antes de la unión.

    Args:
        original_df (pd.DataFrame): El DataFrame original (ej. df_migracion).
        pca_df_with_clusters (pd.DataFrame): El DataFrame que contiene los datos
                                             usados para clustering y la columna de etiquetas.
        cluster_col_name (str): El nombre de la columna de etiquetas en `pca_df_with_clusters`.
        new_col_name (str): El nombre que tendrá la columna de cluster en el DataFrame resultante.

    Returns:
        pd.DataFrame: Una copia del DataFrame original con la columna de cluster añadida.
    """
    print("Asegurando alineación de índices y añadiendo etiquetas de cluster...")
    
    # Trabajar con copias para evitar modificar los dataframes originales
    df_orig_aligned = original_df.reset_index(drop=True)
    df_pca_aligned = pca_df_with_clusters.reset_index(drop=True)

    if len(df_orig_aligned) != len(df_pca_aligned):
        raise ValueError(
            "Las longitudes de los DataFrames no coinciden. "
            f"Original: {len(df_orig_aligned)}, Con Clusters: {len(df_pca_aligned)}"
        )
    
    if cluster_col_name not in df_pca_aligned.columns:
        raise KeyError(f"La columna '{cluster_col_name}' no se encuentra en el DataFrame de PCA.")

    df_result = df_orig_aligned.copy()
    df_result[new_col_name] = df_pca_aligned[cluster_col_name]
    
    print(f"Columna '{cluster_col_name}' añadida como '{new_col_name}'.")
    return df_result