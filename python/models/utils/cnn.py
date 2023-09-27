# External libraries
import matplotlib.pyplot as plt
import pandas as pd

from keras.callbacks import History
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.preprocessing.image import DataFrameIterator, ImageDataGenerator


def create_cnn_model(input_shape: tuple = (128, 128, 1)) -> Sequential:
    """Crea y compila un modelo de red neuronal convolucional (CNN).

    Args:
        input_shape: Tamaño de la entrada (altura, ancho, canales).

    Returns:
        Sequential: Modelo de CNN creado y compilado.

    """
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']
    )

    return model


def image_generator(data: str) -> ImageDataGenerator:
    """Crea y devuelve un generador de datos de imágenes configurado según el
        modo especificado.

    Args:
        data: El modo de los datos, 'train' para datos de entrenamiento o
            'test' para datos de prueba.

    Returns:
        Generador de datos de imágenes configurado según el modo especificado.

    Raises:
        NotImplementedError: Si el valor de 'data' no es 'train' ni 'test'.

    """
    if data == 'train':
        datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,  # Reescala los valores de píxeles a [0, 1]
            rotation_range=20,  # Rotación aleatoria de la imagen
            width_shift_range=0.2,  # Cambio aleatorio en el ancho
            height_shift_range=0.2,  # Cambio aleatorio en la altura
            horizontal_flip=True,  # Volteo horizontal aleatorio
            fill_mode='nearest',  # Modo de relleno para aumentar el tamaño
        )
    elif data == 'test':
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    else:
        raise NotImplementedError('Solo se acepta train o test')

    return datagen


def flow_generator(
    data: pd.DataFrame, datagen: ImageDataGenerator, target: tuple, batch: int
) -> DataFrameIterator:
    """Crea y devuelve un generador de flujo de datos de imágenes a partir de un
        DataFrame.

    Args:
        data: DataFrame que contiene información sobre las imágenes y sus
            etiquetas.
        datagen: Generador de datos de imágenes configurado.
        target: Tamaño objetivo de las imágenes (altura, ancho).
        batch: Tamaño del lote (batch size).

    Returns:
        Generador de flujo de datos de imágenes configurado según las
            especificaciones.

    """
    generator = datagen.flow_from_dataframe(
        dataframe=data,
        x_col='path',  # Nombre de la columna con las rutas de las imágenes
        y_col='label',  # Nombre de la columna con las etiquetas
        target_size=target,  # Tamaño objetivo de las imágenes
        batch_size=batch,  # Tamaño del lote (batch size)
        class_mode='binary',
    )

    return generator


def plot_generate(hist: History, path_save: str = None) -> None:
    """Visualiza las curvas de precisión y pérdida del historial de
        entrenamiento y validación.

    Args:
        hist: Objeto History que contiene el historial de entrenamiento.
        path_save: Ruta para guardar la imagen.

    """
    # Datos del historial de entrenamiento
    train_accuracy = hist.history['accuracy']
    val_accuracy = hist.history['val_accuracy']
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    # Crear una figura con un subplot de 1x2
    plt.figure(figsize=(12, 5))

    # Subplot para la precisión (accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Subplot para la pérdida (loss)
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)

    plt.show()
