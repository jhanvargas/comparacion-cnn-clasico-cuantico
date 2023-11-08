# External libraries
from io import BytesIO
from PIL import Image


def read_image_file(file) -> Image.Image:
    """Lee un archivo de imagen desde datos binarios y lo devuelve como una 
     imagen PIL.

    Args:
        file: Datos binarios de la imagen.

    Returns:
        Image.Image: Un objeto de imagen PIL.

    """
    image_stream = BytesIO(file)
    img = Image.open(image_stream).convert("RGB")
    return img
