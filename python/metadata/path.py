import os


class Path:
    """Paths required for project execution."""

    data = os.path.join('..', '..', 'data')

    qnn_model_example = os.path.join('..', '..', 'output', 'model_train.pt')
