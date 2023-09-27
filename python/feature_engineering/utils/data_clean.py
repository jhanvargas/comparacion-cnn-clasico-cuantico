# External libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from skimage import io
from skimage.transform import resize

# Own libraries
from python.metadata.path import Path


def create_table(
    images_path: str, save: bool = False, target_size: tuple = (128, 128)
) -> pd.DataFrame:
    """Crea un DataFrame a partir de imágenes en las carpetas "other" y
        "portrait" en la ruta especificada.

    Args:
        images_path: La ruta principal que contiene las carpetas "other" y
            "portrait" con las imágenes.
        save: Si es True, guarda el DataFrame en formato parquet.
        target_size: Tamaño al que se redimensionarán las imágenes.

    Returns:
        DataFrame con las imágenes aplanadas y las etiquetas.

    """
    folders = os.listdir(images_path)

    dic = {}
    for folder in folders:
        name, extension = os.path.splitext(folder)
        if extension == '':
            dic[name] = os.path.join(images_path, name)

    image = []
    label = []

    for filename in os.listdir(dic['other']):
        image_path = os.path.join(dic['other'], filename)
        image.append(resize(io.imread(image_path), target_size))
        label.append(0)

    for filename in os.listdir(dic['portrait']):
        image_path = os.path.join(dic['portrait'], filename)
        image.append(resize(io.imread(image_path), target_size))
        label.append(1)

    images = np.array(image)
    labels = np.array(label)

    images = images.reshape(-1, 128 * 128)

    df = pd.DataFrame(images)

    df['label'] = labels

    if save:
        df.to_parquet(Path.portrait_data, index=False)
    else:
        return df


def split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_split: float = 0.2,
    test_split: float = 0.1,
    random_state: int = None,
) -> tuple:
    """Divide los datos de entrenamiento y prueba en conjuntos de entrenamiento,
        validación y prueba de forma aleatoria.

    Args:
        train_df: DataFrame con datos de entrenamiento.
        test_df: DataFrame con datos de prueba.
        validation_split: Proporción de datos de entrenamiento para usar como
            conjunto de validación.
        test_split: Proporción de datos de prueba para usar como conjunto de
            validación.
        random_state: Semilla para la generación de números aleatorios.

    Returns:
        Una tupla que contiene tres DataFrames:
            (train_data, validation_data, test_data).

    """
    train_data, validation_data = train_test_split(
        train_df, test_size=validation_split, random_state=random_state
    )

    if test_split > 0:
        test_data, validation_data = train_test_split(
            test_df, test_size=test_split, random_state=random_state
        )
    else:
        test_data = test_df

    return train_data, validation_data, test_data


def show_random_image(path: str) -> None:
    """Muestra una imagen aleatoria de una carpeta especificada.

    Args:
        path: La ruta de la carpeta que contiene las imágenes.

    """
    files = os.listdir(path)

    random_image = random.choice(files)

    im = io.imread(os.path.join(path, random_image))
    print(im.shape)

    plt.imshow(im)
    plt.title(f'Imagen aleatoria: {random_image}')
    plt.axis('off')
    plt.show()
