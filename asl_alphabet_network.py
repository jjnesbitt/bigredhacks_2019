import warnings  # NOQA

warnings.simplefilter(action="ignore", category=FutureWarning)  # NOQA

import os
import shutil
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


def extract_data(root):
    dirs = [os.path.join(root, p) for p in os.listdir(root)]

    for d in dirs:
        files = os.listdir(d)

        for file in files:
            dest = os.path.join(root, file)
            shutil.copyfile(os.path.join(d, file), dest)


@click.command()
@click.option(
    "--train", is_flag=True, help="Trains the CNN with the provided training data"
)
@click.option("--test", is_flag=True, help="Tests the CNN on the provided testing")
@click.option(
    "--extract",
    is_flag=True,
    help="Runs a script to extract the data into required folder heirarchy",
)
@click.option(
    "-m", "--model", help="The trained model to perform testing or resume training on"
)
@click.option("--data-dir", help="The directory containing the data")
def main(train, test, extract, model, data_dir):
    if (train or test or extract) and not data_dir:
        raise click.UsageError("Data directory not specified")

    if train:
        train()

    elif test:
        test()

    elif extract:
        extract_data(data_dir)


if __name__ == "__main__":
    main()
