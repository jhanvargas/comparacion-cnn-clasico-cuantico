# Own libraries
from python.models.cnn.tensorflow import tensorflow_model
from python.models.cnn.torch import torch_model
from python.models.cnn.hybrid import hybrid_model


def executor():
    tensorflow_model()

    torch_model()

    hybrid_model()
