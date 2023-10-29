import os


class Path:
    """Rutas requeridas para ejecución del proyecto."""

    best_weights_tf = os.path.join('output', 'best_weights_classic_tf.hdf5')
    """Archivo com mejores pesos en CNN clásica con Tensorflow."""

    classic_model_tf = os.path.join('output', 'classic_cnn_tf.h5')
    """Archivo con modelo clásico usando Tensorflow."""

    classic_model_torch = os.path.join('output', 'classic_cnn_torch.pth')
    """Archivo com modelo clásico usando PyTorch."""

    cnn_hybrid_plot = os.path.join('output', 'plot_hybrid_cnn.png')
    """Imagen con comportamiento de entrenamiento y validación híbrido."""

    cnn_tf_plot = os.path.join('output', 'plot_tf_cnn.png')
    """Imagen con comportamiento de entrenamiento y validación clásico tf."""

    cnn_torch_plot = os.path.join('output', 'plot_torch_cnn.png')
    """Imagen con comportamiento de entrenamiento y validación clásico."""

    config = os.path.join('config.yaml')
    """Archivo de configuración de pipelines."""

    data = os.path.join('..', '..', 'data')
    """Ruta de data desde modulo de ejemplos."""

    google_config = os.path.join('input', 'google_drive.json')
    """Archivo de configuración de google drive."""

    hybrid_model_torch = os.path.join('output', 'hybrid_cnn_torch.pth')
    """Archivo com modelo clásico."""

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

    q_circuit = os.path.join('output', 'q_circuit.png')
    """Diagrama del circuito cuántico."""

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
