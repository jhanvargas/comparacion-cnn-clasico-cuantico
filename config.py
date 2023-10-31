# External libraries
import torch

from keras.models import load_model
from functools import lru_cache

# Own libraries
from python.metadata.path import Path
from python.models.utils.torch_cnn import TorchCNN


@lru_cache()
def get_tf_model():
    model = load_model(Path.classic_model_tf)
    model.load_weights(Path.best_weights_tf)
    return model


@lru_cache()
def get_pyt_model():
    model = TorchCNN()
    model.load_state_dict(torch.load(Path.classic_model_torch))
    return model
