# Own libraries
from python.models.cnn.tensorflow import tensorflow_model
from python.models.cnn.torch import torch_model
from python.models.cnn.hybrid import hybrid_model


def executor() -> None:
    """Ejecuta pipelines para modelos de clasificación de imágenes.

    Esta función llama a tres modelos diferentes para la clasificación de 
    imágenes:
        - tensorflow_model(): Un modelo CNN utilizando TensorFlow.
        - torch_model(): Un modelo CNN utilizando PyTorch.
        - hybrid_model(): Un modelo CNN utilizando PyTorch y Qiskit.

    """
    tensorflow_model()
    torch_model()
    hybrid_model()
