# Own libraries
from python.feature_engineering import feature_pipeline
from python.models import neural_networks


if __name__ == '__main__':
    feature_pipeline.executor()
    """Pipeline de feature engineering."""

    neural_networks.executor()
    """Pipeline de modelos."""
