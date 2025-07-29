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

