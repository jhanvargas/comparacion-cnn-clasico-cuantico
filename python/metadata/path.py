import os


class Path:
    """Paths required for project execution."""

    data = os.path.join('..', '..', 'data')

    config = os.path.join('input', 'config.yaml')

    kaggle_config = os.path.join('input', 'kaggle.json')

    google_config = os.path.join('input', 'google_drive.json')

    qnn_model_example = os.path.join('..', '..', 'output', 'model_train.pt')

    output = os.path.join('output')

    images = os.path.join('output', 'binary')

    test_binary = os.path.join(images, 'test_binary.csv')

    train_binary = os.path.join(images, 'train_binary.csv')

    train = os.path.join(images, 'train.csv')

    test = os.path.join(images, 'test.csv')

    val = os.path.join(images, 'val.csv')

    best_model = os.path.join('output', 'best_model.hdf5')

    portrait_data = os.path.join('output', 'portrait_data.parquet')
