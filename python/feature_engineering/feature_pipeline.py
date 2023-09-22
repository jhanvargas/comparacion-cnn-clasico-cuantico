# Own libraries
from python.feature_engineering.utils.download import download_dataset
from python.utils.readers import read_yaml
from python.metadata.path import Path


def executor():
    config = read_yaml(Path.config)['dataset']
    download = config['download']

    if download:
        dataset = config['dataset_name']
        download_dataset(dataset_name=dataset, output_path=Path.output)
