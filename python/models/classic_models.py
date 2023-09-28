# External libraries
import pandas as pd

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


def executor():
    """Pipeline de modelo de CNN clásica."""

    config = read_yaml(Path.config)['cnn_classic']
    train = config['train']
    test = config['test']

    batch = 32
    target = (32, 32)

    if train:
        train = pd.read_csv(Path.train, converters={'label': str})
        val = pd.read_csv(Path.val, converters={'label': str})

        train_datagen = image_generator('train')
        test_datagen = image_generator('test')

        train_generator = flow_generator(train, train_datagen, target, batch)
        val_generator = flow_generator(val, test_datagen, target, batch)

        checkpoint = ModelCheckpoint(
            Path.best_weights,
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
            epochs=10,
            callbacks=[checkpoint],
        )

        model.save(Path.classic_model)

        plot_generate(hist, Path.cnn_classic_plot)

    if test:
        test = pd.read_csv(Path.test, converters={'label': str})
        test_datagen = image_generator('test')
        test_generator = flow_generator(test, test_datagen, target, batch)

        model = load_model(Path.classic_model)
        model.load_weights(Path.best_weights)

        model.evaluate(test_generator)
