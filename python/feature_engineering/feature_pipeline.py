# External libraries
import pandas as pd

# Own libraries
from python.feature_engineering.utils.data_clean import split_data
from python.feature_engineering.utils.download import download_dataset
from python.metadata.path import Path
from python.utils.readers import read_yaml


def executor():
    """Pipeline para la obtención y organización de datos."""

    config = read_yaml(Path.config)['dataset']
    download = config['download']
    split = config['split_data']

    if download:
        dataset = config['dataset_name']
        download_dataset(dataset_name=dataset, output_path=Path.output)

    if split:
        train_binary = pd.read_csv(Path.train_binary)
        test_binary = pd.read_csv(Path.test_binary)

        train_binary['x:image'] = train_binary['x:image'].apply(
            lambda x: Path.images + x[1:]
        )

        test_binary['x:image'] = test_binary['x:image'].apply(
            lambda x: Path.images + x[1:]
        )

        renames = {'x:image': 'path', 'y:label': 'label'}
        train_binary.rename(columns=renames, inplace=True)
        test_binary.rename(columns=renames, inplace=True)

        train_data, validation_data, test_data = split_data(
            train_binary,
            test_binary,
            validation_split=0.2,
            test_split=0,
            random_state=42,
        )

        train_data.to_csv(Path.train, index=False)
        validation_data.to_csv(Path.val, index=False)
        test_data.to_csv(Path.test, index=False)
