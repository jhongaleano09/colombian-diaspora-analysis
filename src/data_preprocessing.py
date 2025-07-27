import pandas as pd
import numpy as np

def load_data(file_path):
    """Carga el dataset desde una ruta de archivo CSV."""
    print(f"Cargando dataset desde: {file_path}")
    df = pd.read_csv(file_path)
    print("¡Dataset cargado exitosamente!")
    return df

def clean_data(df):
    """
    Realiza la limpieza inicial de datos: convierte fechas, maneja valores
    no estándar y elimina registros inconsistentes.
    """
    print("Iniciando limpieza de datos...")
    # Convertir fecha de registro y extraer componentes
    df['Fecha de Registro'] = pd.to_datetime(df['Fecha de Registro'], errors='coerce')
    df.dropna(subset=['Fecha de Registro'], inplace=True) # Eliminar filas donde la fecha no se pudo convertir
    df['Año Registro'] = df['Fecha de Registro'].dt.year

    # Eliminar registros anómalos del año 1900
    df = df[df['Año Registro'] != 1900].copy()

    # Reemplazar valores no estándar en 'Edad (años)' con la media
    mean_age_valid = df[df['Edad (años)'] != -1]['Edad (años)'].mean()
    df.loc[df['Edad (años)'] == -1, 'Edad (años)'] = mean_age_valid

    # Reemplazar valores no estándar en 'Estatura (CM)'
    # (Asumimos un reemplazo con la media/mediana o un valor fijo)
    df.loc[(df['Estatura (CM)'] < 80) | (df['Estatura (CM)'] > 220), 'Estatura (CM)'] = 165

    print(f"Limpieza finalizada. Forma del DataFrame: {df.shape}")
    return df

def preprocess_features(df):
    """Prepara las características para el modelado (encoding, scaling, etc.)."""
    # Aquí irían tus funciones para One-Hot Encoding, Label Encoding, StandardScaler, etc.
    # Ejemplo:
    # df_encoded = pd.get_dummies(df, columns=['País', 'Nivel Académico'])
    # ...
    return df_processed