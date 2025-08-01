from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(X, y):
    """Entrena un modelo Random Forest para predecir el nivel académico."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy del Random Forest: {accuracy:.3f}")
    print(classification_report(y_test, y_pred))

    return model


# src/supervised_modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import lightgbm as lgb
from typing import Dict, Any, List, Optional

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Divide los datos en conjuntos de entrenamiento y prueba de forma estratificada."""
    print(f"Dividiendo datos: {1-test_size:.0%} entrenamiento, {test_size:.0%} prueba.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Tamaño Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_classifier(X_train: pd.DataFrame, y_train: pd.Series, features: list, model_type: str, model_params: Optional[Dict] = None) -> Optional[Pipeline]:
    """
    Crea y entrena un pipeline de clasificación.

    Args:
        X_train, y_train: Datos de entrenamiento.
        features (list): Lista de columnas a escalar.
        model_type (str): Tipo de modelo ('rf' para RandomForest, 'lgbm' para LightGBM).
        model_params (dict, optional): Hiperparámetros para el clasificador.

    Returns:
        Pipeline: El pipeline entrenado o None si hay un error.
    """
    if model_params is None:
        model_params = {}

    preprocessor = ColumnTransformer(
        transformers=[('scaler', StandardScaler(), features)],
        remainder='drop'
    )

    classifiers = {
        'rf': RandomForestClassifier,
        'lgbm': lgb.LGBMClassifier
    }
    
    if model_type not in classifiers:
        raise ValueError(f"Tipo de modelo '{model_type}' no soportado. Opciones: {list(classifiers.keys())}")
    
    try:
        classifier = classifiers[model_type](**model_params)
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        pipeline.fit(X_train, y_train)
        print(f"Pipeline para '{model_type}' entrenado exitosamente.")
        return pipeline

    except Exception as e:
        print(f"Error al entrenar el modelo '{model_type}': {e}")
        return None

def evaluate_classification_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, Any]:
    """
    Evalúa un pipeline de clasificación y muestra las métricas.

    Args:
        pipeline (Pipeline): El pipeline entrenado.
        X_test, y_test: Datos de prueba.
        model_name (str): Nombre del modelo para los reportes.

    Returns:
        dict: Un diccionario con los resultados de la evaluación.
    """
    print(f"\n--- Resultados de Evaluación: {model_name} ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report_str = classification_report(y_test, y_pred)
    
    # Calcular AUC ROC
    unique_labels = sorted(y_test.unique())
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted', labels=unique_labels)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC ROC (Weighted OvR): {auc:.4f}")
    except ValueError as e:
        auc = None
        print(f"Accuracy: {accuracy:.4f}")
        print(f"No se pudo calcular AUC ROC: {e}")

    print("\nClassification Report:")
    print(report_str)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'auc': auc,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'predictions': y_pred
    }