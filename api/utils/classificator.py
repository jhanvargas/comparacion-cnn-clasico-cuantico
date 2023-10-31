# External libraries
import numpy as np
import torch

from torchvision import transforms

# Own libraries
from config import get_tf_model, get_pyt_model


def predict(img) -> dict:

    classes = {0: 'Other', 1: 'Portrait'}

    im = img.copy().convert("L")
    img = img.resize((32, 32))
    img_array = np.array(img, dtype='float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    tf_model = get_tf_model()
    prediction = tf_model.predict(img_array)[0][0]
    # predicted_tf = np.argmax(int(prediction), axis=-1)
    predicted_tf = 0 if prediction < 0.5 else 1

    pyt_model = get_pyt_model()

    preprocess = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    im = preprocess(im)
    im = torch.unsqueeze(im, 0)

    with torch.no_grad():
        predictions = pyt_model(im)
        # predicted_pyt = torch.argmax(predictions, dim=1).item()

    predicted_pyt = 0 if predictions < 0.5 else 1

    return {
        'Tensorflow model predict': f'The image is a {classes[predicted_tf]}',
        'PyTorch model predict': f'The image is a {classes[predicted_pyt]}',
    }
