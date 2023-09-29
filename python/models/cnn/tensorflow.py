# External libraries
import pandas as pd
import tensorflow as tf

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Own libraries
from python.metadata.path import Path
from python.models.utils.tf_cnn import (
    create_tf_cnn,
    flow_generator,
    image_generator,
    plot_generate,
)
from python.utils.readers import read_yaml


def tensorflow_model():
    """Pipeline de modelo de CNN clásica con tensorflow."""

    if tf.config.experimental.list_physical_devices('GPU'):
        print('TensorFlow está utilizando la GPU.')
    else:
        print('TensorFlow no está utilizando la GPU.')

    config = read_yaml(Path.config)['cnn_models']['tf_cnn_classic']
    train = config['train']
    test = config['test']

    batch = config['batch_size']
    target = tuple(config['input_target'])
    epochs = config['epochs']

    if train:
        train = pd.read_csv(Path.train, converters={'label': str})
        val = pd.read_csv(Path.val, converters={'label': str})

        train_datagen = image_generator('train')
        test_datagen = image_generator('test')

        train_generator = flow_generator(train, train_datagen, target, batch)
        val_generator = flow_generator(val, test_datagen, target, batch)

        checkpoint = ModelCheckpoint(
            Path.best_weights_tf,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
        )

        model = create_tf_cnn(input_shape=(32, 32, 3))
        print(model.summary())

        hist = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch,
            validation_data=val_generator,
            validation_steps=val_generator.samples // batch,
            epochs=epochs,
            callbacks=[checkpoint],
        )

        model.save(Path.classic_model_tf)

        plot_generate(hist, Path.cnn_tf_plot)

    if test:
        test = pd.read_csv(Path.test, converters={'label': str})
        test_datagen = image_generator('test')
        test_generator = flow_generator(test, test_datagen, target, batch)

        model = load_model(Path.classic_model_tf)
        model.load_weights(Path.best_weights_tf)

        model.evaluate(test_generator)
