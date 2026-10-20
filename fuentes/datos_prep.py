
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def cargar_datos(ruta_archivo):
    """
    Carga un archivo CSV y devuelve una tabla de datos.
    """
    tabla1 = pd.read_csv(ruta_archivo)
    return tabla1


def limpiar_datos(tabla1):
    """
    Elimina identificadores y codifica la variable objetivo.
    """
    tabla_limpia = tabla1.copy()

    columnas_eliminar = ["Unnamed: 0", "id"]

    tabla_limpia = tabla_limpia.drop(
        columns=columnas_eliminar,
        errors="ignore"
    )

    if "satisfaction" not in tabla_limpia.columns:
        raise ValueError(
            "No se encontró la variable objetivo satisfaction."
        )

    tabla_limpia["satisfaction"] = (
        tabla_limpia["satisfaction"].map({
            "neutral or dissatisfied": 0,
            "satisfied": 1
        })
    )

    if tabla_limpia["satisfaction"].isnull().any():
        raise ValueError(
            "Se encontraron valores no reconocidos en satisfaction."
        )

    tabla_limpia["satisfaction"] = (
        tabla_limpia["satisfaction"].astype(int)
    )

    return tabla_limpia


def separar_variables(tabla_limpia):
    """
    Separa las variables predictoras y la variable objetivo.
    """
    datos_principales = tabla_limpia.drop(
        columns=["satisfaction"]
    )

    valor_principal = tabla_limpia["satisfaction"]

    return datos_principales, valor_principal


def crear_transformador(datos_principales):
    """
    Configura las transformaciones de variables
    numéricas y categóricas.
    """
    columnas_numericas = datos_principales.select_dtypes(
        include=["number"]
    ).columns.tolist()

    columnas_categoricas = datos_principales.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    proceso_numerico = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="median")),
        ("escalado", StandardScaler())
    ])

    proceso_categorico = Pipeline(steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("codificacion", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformador_principal = ColumnTransformer(
        transformers=[
            ("numericas", proceso_numerico, columnas_numericas),
            ("categoricas", proceso_categorico, columnas_categoricas)
        ]
    )

    return transformador_principal


def preparar_datos(ruta_archivo):
    """
    Ejecuta la preparación inicial y devuelve los
    datos y el transformador sin ajustar.
    """
    tabla1 = cargar_datos(ruta_archivo)

    tabla_limpia = limpiar_datos(tabla1)

    datos_principales, valor_principal = separar_variables(
        tabla_limpia
    )

    transformador_principal = crear_transformador(
        datos_principales
    )

    return (
        datos_principales,
        valor_principal,
        transformador_principal
    )


if __name__ == "__main__":

    import argparse

    analizador = argparse.ArgumentParser(
        description="Preparación de datos de satisfacción de pasajeros"
    )

    analizador.add_argument(
        "--entrada",
        default="datos/datos_ini/train.csv",
        help="Ruta del archivo CSV original"
    )

    argumentos = analizador.parse_args()

    datos_principales, valor_principal, transformador_principal = (
        preparar_datos(argumentos.entrada)
    )

    print("PREPARACIÓN DE DATOS")
    print("Registros:", len(datos_principales))
    print("Variables predictoras:", datos_principales.shape[1])
    print("Variable objetivo:", valor_principal.name)
    print("Valores nulos:", datos_principales.isnull().sum().sum())
    print("Transformador configurado correctamente.")
# Funciones para la limpieza y preparación de datos
