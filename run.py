# Own libraries
from python.feature_engineering import feature_pipeline
from python.feature_engineering.utils.data_clean import show_random_image
from python.metadata.path import Path
from python.models import classic_models
from python.ibm_quantum.utils.connect import get_ibm_quantum


if __name__ == '__main__':
    feature_pipeline.executor()
    """Pipeline de feature engineering."""

    show_random_image(Path.portrait)

    classic_models.executor()
    """Pipeline de modelos."""
