
# 12. Entrenamiento, evaluación y registro de modelos en MLflow

import os
import argparse
import tempfile

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from datos_prep import preparar_datos, cargar_datos, limpiar_datos, separar_variables


def ejecutar_entrenamiento(ruta_entrenamiento, ruta_prueba, ruta_seguimiento):

    # Configurar MLflow
    mlflow.set_tracking_uri(ruta_seguimiento)
    mlflow.set_experiment("Actividad5_Clasificacion_Pasajeros")

    # Cargar y preparar los datos de entrenamiento
    datos_entrenamiento, valores_entrenamiento, transformador_principal = (
        preparar_datos(ruta_entrenamiento)
    )

    # Cargar y preparar los datos de prueba
    tabla_prueba = cargar_datos(ruta_prueba)
    tabla_prueba = limpiar_datos(tabla_prueba)

    datos_prueba, valores_prueba = separar_variables(tabla_prueba)

    # Verificar consistencia entre ambos conjuntos
    if set(datos_entrenamiento.columns) != set(datos_prueba.columns):
        raise ValueError(
            "Las variables de entrenamiento y prueba no coinciden."
        )

    datos_prueba = datos_prueba[datos_entrenamiento.columns]

    # Definir algoritmos con los hiperparámetros optimizados
    modelos_principales = {
        "Regresion_Logistica": LogisticRegression(
            C=10,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
    }

    # Configurar validación cruzada
    validacion_principal = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Definir métricas
    metricas_principales = {
        "exactitud": "accuracy",
        "precision": "precision",
        "sensibilidad": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    resultados_principales = []

    # Entrenar y evaluar cada algoritmo
    for nombre_modelo, algoritmo_principal in modelos_principales.items():

        print("\nENTRENANDO:", nombre_modelo)

        modelo_principal = Pipeline(steps=[
            ("preparacion", clone(transformador_principal)),
            ("clasificacion", clone(algoritmo_principal))
        ])

        # Registrar experimento
        with mlflow.start_run(run_name=nombre_modelo + "_Script"):

            mlflow.log_param("algoritmo", nombre_modelo)
            mlflow.log_param("particiones", 5)
            mlflow.log_param("semilla", 42)
            mlflow.log_param("registros_entrenamiento", len(datos_entrenamiento))
            mlflow.log_param("registros_prueba", len(datos_prueba))

            # Registrar hiperparámetros
            for parametro, valor in algoritmo_principal.get_params().items():
                if isinstance(valor, (str, int, float, bool)) or valor is None:
                    mlflow.log_param(
                        "modelo_" + parametro,
                        str(valor)
                    )

            # Ejecutar validación cruzada
            resultados_validacion = cross_validate(
                modelo_principal,
                datos_entrenamiento,
                valores_entrenamiento,
                cv=validacion_principal,
                scoring=metricas_principales,
                n_jobs=2,
                error_score="raise"
            )

            # Registrar métricas de validación
            for nombre_metrica in metricas_principales:

                valores_metrica = resultados_validacion[
                    "test_" + nombre_metrica
                ]

                mlflow.log_metric(
                    nombre_metrica + "_validacion",
                    float(np.mean(valores_metrica))
                )

                mlflow.log_metric(
                    nombre_metrica + "_desviacion",
                    float(np.std(valores_metrica))
                )

            # Entrenar con el conjunto completo de entrenamiento
            modelo_principal.fit(
                datos_entrenamiento,
                valores_entrenamiento
            )

            # Evaluar con datos de prueba
            predicciones = modelo_principal.predict(datos_prueba)

            probabilidades = modelo_principal.predict_proba(
                datos_prueba
            )[:, 1]

            resultado_modelo = {
                "Modelo": nombre_modelo,
                "Exactitud": accuracy_score(valores_prueba, predicciones),
                "Precision": precision_score(valores_prueba, predicciones),
                "Sensibilidad": recall_score(valores_prueba, predicciones),
                "F1": f1_score(valores_prueba, predicciones),
                "ROC_AUC": roc_auc_score(valores_prueba, probabilidades)
            }

            # Registrar métricas finales
            for nombre_metrica, valor_metrica in resultado_modelo.items():
                if nombre_metrica != "Modelo":
                    mlflow.log_metric(
                        nombre_metrica.lower() + "_test",
                        float(valor_metrica)
                    )

            # Registrar modelo completo en MLflow
            mlflow.sklearn.log_model(
                sk_model=modelo_principal,
                name="modelo_clasificacion",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.compose._column_transformer._RemainderColsList",
                    "sklearn.tree._tree.Tree"
                ]
            )

            resultados_principales.append(resultado_modelo)

            print("Entrenamiento y evaluación finalizados.")
            print("F1:", round(resultado_modelo["F1"], 4))
            print("ROC-AUC:", round(resultado_modelo["ROC_AUC"], 4))
            print("Resultados registrados en MLflow.")

    # Generar comparación final
    tabla_resultados = pd.DataFrame(resultados_principales)

    print("\nCOMPARACIÓN FINAL DE MODELOS")
    print(tabla_resultados.round(4).to_string(index=False))

    # Guardar comparación como artefacto
    with tempfile.TemporaryDirectory() as ruta_temporal:

        ruta_resultados = os.path.join(
            ruta_temporal,
            "comparacion_modelos.csv"
        )

        tabla_resultados.to_csv(ruta_resultados, index=False)

        with mlflow.start_run(run_name="Comparacion_Script"):
            mlflow.log_artifact(ruta_resultados)

    print("\nProceso completado correctamente.")

    return tabla_resultados


if __name__ == "__main__":

    analizador = argparse.ArgumentParser(
        description="Entrenamiento y registro de modelos en MLflow"
    )

    analizador.add_argument(
        "--entrenamiento",
        default="datos/datos_ini/train.csv",
        help="Ruta del archivo de entrenamiento"
    )

    analizador.add_argument(
        "--prueba",
        default="datos/datos_ini/test.csv",
        help="Ruta del archivo de prueba"
    )

    analizador.add_argument(
        "--mlflow",
        default="sqlite:///mlflow.db",
        help="Ubicación de la base de datos de MLflow"
    )

    argumentos = analizador.parse_args()

    ejecutar_entrenamiento(
        argumentos.entrenamiento,
        argumentos.prueba,
        argumentos.mlflow
    )
# Script de entrenamiento y registro de modelos en MLflow
