import os


class Path:
    """Rutas requeridas para ejecución del proyecto."""

    best_weights = os.path.join('output', 'best_weights_classic_100.hdf5')
    """Archivo com mejores pesos en CNN clásica."""

    classic_model = os.path.join('output', 'classic_cnn.h5')
    """Archivo com modelo clásico."""

    classic_model_torch = os.path.join('output', 'classic_cnn_torch.pth')
    """Archivo com modelo clásico."""

    cnn_classic_plot = os.path.join('output', 'classic_cnn.png')
    """Imagen con comportamiento de entrenamiento y validación clásico."""

    cnn_torch_plot = os.path.join('output', 'cnn_torch.png')
    """Imagen con comportamiento de entrenamiento y validación clásico."""

    config = os.path.join('input', 'config.yaml')
    """Archivo de configuración de pipelines."""

    data = os.path.join('..', '..', 'data')
    """Ruta de data desde modulo de ejemplos."""

    google_config = os.path.join('input', 'google_drive.json')
    """Archivo de configuración de google drive."""

    ibm_config = os.path.join('input', 'ibm_quamtum.yaml')
    """API key para IBM Quantum."""

    images = os.path.join('output', 'binary')
    """Ruta a carpeta con imágenes del proyecto."""

    kaggle_config = os.path.join('input', 'kaggle.json')
    """Archivo de configuración de kaggle."""

    output = os.path.join('output')
    """Ruta a carpeta de salida."""

    portrait = os.path.join('output', 'binary', 'portrait')
    """Ruta a imágenes de retratos."""

    portrait_data = os.path.join('output', 'portrait_data.parquet')
    """Archivo con df de imágenes en arreglo."""

    qnn_model_example = os.path.join('..', '..', 'output', 'model_train.pt')
    """Modelo de qnn de ejemplos."""

    test = os.path.join(images, 'test.csv')
    """Archivo con datos de imágenes de testeo."""

    test_binary = os.path.join(images, 'test_binary.csv')
    """Ruta de df de testeo."""

    train = os.path.join(images, 'train.csv')
    """Archivo con datos de imágenes de entrenamiento."""

    train_binary = os.path.join(images, 'train_binary.csv')
    """Ruta de df de entrenamiento."""

    val = os.path.join(images, 'val.csv')
    """Archivo con datos de imágenes de validación."""
