# External libraries
import json
import os
import yaml


def set_environ_kaggle(path: str) -> None:
    """Establece las variables de entorno para la autenticación
     de la API de Kaggle.

    Args:
        path: Ruta al archivo kaggle.json que contiene las
         credenciales de API de Kaggle.

    """
    with open(path, 'r') as f:
        data = json.load(f)

    os.environ['KAGGLE_USERNAME'] = data['username']
    os.environ['KAGGLE_KEY'] = data['key']


def read_yaml(path: str) -> dict:
    """Lee y analiza datos YAML desde un archivo especificado.

    Args:
        path: La ruta al archivo YAML que se va a leer.

    Returns:
        Un diccionario que contiene los datos YAML analizados.

    """
    with open(path, encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data
