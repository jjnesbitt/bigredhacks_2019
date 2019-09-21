import click
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard


def get_model() -> Sequential:
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(200, 200, 1)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(29, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )


def train(x_train, y_train, batch_size=None, epochs=1, validation_split=0.0, callbacks=None):
    model = get_model()
    model.fit(x_train, y_train, batch_size, epochs, validation_split)


def test(x_test, y_train):
    pass


@click.command()
@click.option("--train", help="Trains the CNN with the provided training data")
@click.option("--test", help="Tests the CNN on the provided testing")
@click.option("--extract-data")
@click.option("-m", "--model")
@click.option("--training-data")
@click.option("--testing-data")
def main(train, test, model, data_dir):
    if train:
        if not data_dir:
            click.UsageError("Training data directory not specified")

        train()

    elif test:
        test()


if __name__ == "__main__":
    main()
