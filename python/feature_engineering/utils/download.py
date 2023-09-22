# External libraries
import os
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

# Own libraries
from python.metadata.path import Path
from python.utils.readers import set_environ_kaggle


def download_dataset(dataset_name: str, output_path: str) -> None:
    """Descarga un conjunto de datos de Kaggle y lo extrae en
     la ubicación especificada.

    Args:
        dataset_name: Nombre del conjunto de datos en Kaggle.
         Debe seguir el formato 'nombre-usuario/nombre-conjunto-datos'.
        output_path: Ruta donde se extraerán los archivos
         del conjunto de datos.

    """
    set_environ_kaggle(Path.kaggle_config)

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(dataset_name, path=output_path, unzip=True)
