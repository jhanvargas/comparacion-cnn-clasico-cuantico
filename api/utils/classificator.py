# External libraries
import numpy as np
import torch

from keras.models import load_model
from torchvision import transforms

# Own libraries
from python.metadata.path import Path
from python.models.utils.torch_cnn import TorchCNN


def predict(img) -> dict:
    im = img.copy().convert("L")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    model = load_model(Path.classic_model_tf)
    model.load_weights(Path.best_weights_tf)

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        classes = 'Is a portrait'
    else:
        classes = 'Is a other'

    model_ = TorchCNN()
    model_.load_state_dict(torch.load(Path.classic_model_torch))

    preprocess = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )

    im = preprocess(im)
    im = torch.unsqueeze(im, 0)

    with torch.no_grad():
        predictions = model_(im)[0][0]

    return {
        'Tensorflow model predict': classes,
        'PyTorch model predict': str(predictions)}

