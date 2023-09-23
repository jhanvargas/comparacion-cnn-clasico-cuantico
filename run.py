# Own libraries
from python.feature_engineering import feature_pipeline
from python.feature_engineering.utils.data_clean import create_table
from python.metadata.path import Path

if __name__ == '__main__':
    feature_pipeline.executor()
    create_table(Path.images, save=True)
