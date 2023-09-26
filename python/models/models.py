# External libraries
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Own libraries
from python.metadata.path import Path
from python.models.utils.cnn import create_cnn_model


def executor():
    train = pd.read_csv(Path.train, converters={'label': str})
    val = pd.read_csv(Path.val, converters={'label': str})
    test = pd.read_csv(Path.test, converters={'label': str})

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Reescala los valores de píxeles a [0, 1]
        rotation_range=20,  # Rotación aleatoria de la imagen
        width_shift_range=0.2,  # Cambio aleatorio en el ancho
        height_shift_range=0.2,  # Cambio aleatorio en la altura
        horizontal_flip=True,  # Volteo horizontal aleatorio
        fill_mode='nearest'  # Modo de relleno para aumentar el tamaño
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        x_col='path',  # Nombre de la columna con las rutas de las imágenes
        y_col='label',  # Nombre de la columna con las etiquetas
        target_size=(128, 128),  # Tamaño objetivo de las imágenes
        batch_size=128,  # Tamaño del lote (batch size)
        class_mode='binary'  # Cambia a 'categorical' si tienes más de dos clases
    )

    val_generator = test_datagen.flow_from_dataframe(
        dataframe=val,
        x_col='path',  # Nombre de la columna con las rutas de las imágenes
        y_col='label',  # Nombre de la columna con las etiquetas
        target_size=(128, 128),  # Tamaño objetivo de las imágenes
        batch_size=128,  # Tamaño del lote (batch size)
        class_mode='binary'  # Cambia a 'categorical' si tienes más de dos clases
    )

    checkpoint = ModelCheckpoint(
        Path.best_model, monitor='val_accuracy', verbose=1, save_best_only=True
    )

    model = create_cnn_model()

    hist = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 128,
        validation_data=val_generator,
        validation_steps=val_generator.samples // 128,
        epochs=10,
        callbacks=[checkpoint]
    )

    plt.plot(hist.history['accuracy'], label='Train')
    plt.plot(hist.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.show()

    #model.load_weights(Path.best_model)
