# Actividad 5. Clasificación de satisfacción de pasajeros

## Descripción del proyecto

Este proyecto tiene como objetivo entrenar y comparar los algoritmos de **Regresión Logística y Random Forest** para predecir la satisfacción de pasajeros de aerolíneas, aplicando técnicas de aprendizaje supervisado y buenas prácticas de MLOps.

## Dataset

Utilizamos el conjunto de datos [Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction), disponible en Kaggle, que contiene información sobre pasajeros, características de vuelos y evaluaciones de servicios.

## Metodología

Realizamos limpieza de datos, imputación de valores nulos, codificación de variables categóricas y estandarización. Posteriormente, entrenamos ambos algoritmos, aplicamos validación cruzada y optimizamos hiperparámetros mediante Grid Search.

Utilizamos **MLflow** para registrar parámetros, métricas, modelos y evidencias de los experimentos.

## Resultados

| Métrica | Regresión Logística | Random Forest |
|---|---:|---:|
| Accuracy | 87.18 % | 96.32 % |
| Precision | 86.94 % | 97.16 % |
| Recall | 83.33 % | 94.38 % |
| F1-score | 85.09 % | 95.75 % |
| ROC-AUC | 92.56 % | 99.40 % |

Random Forest obtuvo el mejor desempeño y presentó diferencias estadísticamente significativas frente a la Regresión Logística, según la prueba de McNemar.

## Estructura del repositorio

- `datos/`: archivos originales y espacio para datos procesados.
- `fuentes/entrena.ipynb`: desarrollo experimental en Google Colab.
- `fuentes/datos_prep.py`: funciones de preparación de datos.
- `fuentes/train.py`: entrenamiento, evaluación y registro en MLflow.
- `evidencias/mlflow/`: resultados y visualizaciones de los experimentos.
- `CHANGELOG.md`: registro de cambios del proyecto.

## Ejecución

Instalar las dependencias:

```bash
pip install pandas numpy scikit-learn mlflow skops
```

Ejecutar desde la raíz del repositorio:

```bash
python fuentes/train.py
```

El script entrena ambos algoritmos, realiza la evaluación y registra los resultados en MLflow.
