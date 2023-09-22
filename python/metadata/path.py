import os


class Path:
    """Paths required for project execution."""
    path_file = os.path.dirname(os.path.abspath(__file__))

    data = os.path.join('..', '..', 'data')

    config = os.path.join('input', 'config.yaml')

    kaggle_config = os.path.join('input', 'kaggle.json')

    qnn_model_example = os.path.join('..', '..', 'output', 'model_train.pt')

    output = os.path.join('output')
