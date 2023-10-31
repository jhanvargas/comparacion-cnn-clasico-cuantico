# External libraries
from io import BytesIO
from PIL import Image

from keras.preprocessing import image


def read_image_file(file) -> Image.Image:
    """Reads an image file from binary data and returns it as a PIL Image.

    Args:
        file: Binary image data.

    Returns:
        Image.Image: A PIL Image object.

    """
    image_stream = BytesIO(file)
    img = Image.open(image_stream).convert("RGB")
    return img
