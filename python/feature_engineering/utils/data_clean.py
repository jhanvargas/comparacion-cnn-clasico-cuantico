# External libraries
import os
import pandas as pd
import numpy as np

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
