# Librerías propias
from python.feature_engineering import feature_pipeline
from python.models import neural_networks


def main() -> None:
    """Ejecuta los pipelines de ingeniería de características y redes neuronales.

    Esta función ejecuta los pipelines de ingeniería de características y redes neuronales
    definidos en los módulos feature_pipeline y neural_networks, respectivamente.

    """
    feature_pipeline.executor()
    neural_networks.executor()


if __name__ == '__main__':
    main()
