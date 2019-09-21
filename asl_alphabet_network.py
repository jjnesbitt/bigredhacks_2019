import click
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten


def get_model() -> Sequential:
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(200, 200, 1)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(29, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )


def train(x_train, y_train, x_test, y_test):
    model = get_model()
    model.fit()


def test():
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
