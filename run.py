# Own libraries
from python.feature_engineering import feature_pipeline
from python.models import models

if __name__ == '__main__':
    feature_pipeline.executor()
    models.executor()
