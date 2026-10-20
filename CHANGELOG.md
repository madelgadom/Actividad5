# CHANGELOG - Actividad 5

## Versión 1.0.0 - Implementación inicial

### Preparación de datos
- Incorporación del dataset *Airline Passenger Satisfaction* de Kaggle.
- Análisis exploratorio y visualización de datos.
- Tratamiento de valores nulos, codificación de variables categóricas y estandarización.

### Entrenamiento y optimización
- Implementación de Regresión Logística y Random Forest.
- Validación cruzada estratificada.
- Optimización de hiperparámetros mediante Grid Search.

### Evaluación de modelos
- Evaluación mediante Accuracy, Precision, Recall, F1 y ROC-AUC.
- Comparación estadística mediante la prueba de McNemar.
- Selección de Random Forest como mejor modelo, con una exactitud del 96.32 %.

### Implementación de MLOps
- Configuración de MLflow para el seguimiento de experimentos.
- Registro de parámetros, métricas, modelos y artefactos.
- Desarrollo de los archivos `datos_prep.py` y `train.py`.
- Incorporación de evidencias y documentación en GitHub.

### Resultado final
Se implementó un proceso reproducible de clasificación de satisfacción de pasajeros, integrando preparación de datos, entrenamiento, evaluación y seguimiento de experimentos mediante MLflow.
