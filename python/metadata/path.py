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

    confusion_matrix_tf = os.path.join('output', 'confusion_matrix_tf.png')
    """Imagen con matriz de confusión de CNN clásica con Tensorflow."""

    confusion_matrix_torch = os.path.join('output', 'confusion_matrix_torch.png')
    """Imagen con matriz de confusión de CNN clásica con PyTorch."""

    confusion_matrix_hybrid = os.path.join('output', 'confusion_matrix_hy.png')
    """Imagen con matriz de confusión de CNN híbrida."""

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

    images_grid_other = os.path.join('output', 'grid_other.png')
    """Imagen con grid de imágenes de otros."""

    images_grid_portrait = os.path.join('output', 'grid_portrait.png')
    """Imagen con grid de imágenes de retratos."""

    intensities = os.path.join('output', 'intensities.png')
    """Imagen con intensidades de imágenes de retratos."""

    kaggle_config = os.path.join('input', 'kaggle.json')
    """Archivo de configuración de kaggle."""

    label_distribution = os.path.join('output', 'label_distribution.png')
    """Imagen con distribución de etiquetas."""

    other = os.path.join('output', 'binary', 'other')
    """Ruta a imágenes de otros."""

    output = os.path.join('output')
    """Ruta a carpeta de salida."""

    pca = os.path.join('output', 'pca.png')
    """Imagen con PCA de imágenes de retratos."""

    pixel_df = os.path.join('output', 'pixel_df.parquet')
    """Archivo con df de imágenes en arreglo."""

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
