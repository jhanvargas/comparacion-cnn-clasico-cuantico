# Own libraries
from python.models.cnn.tensorflow import tensorflow_model
from python.models.cnn.torch import torch_model
from python.models.cnn.hybrid import hybrid_model


def executor():
    """Pipelines de los modelos de clasificación de imágenes."""

    tensorflow_model()
    """Modelo de CNN usando tensowflow."""

    torch_model()
    """Modelo de CNN usando PyTorch."""

    hybrid_model()
    """Modelo de CNN usando PyTorch y Qiskit."""
