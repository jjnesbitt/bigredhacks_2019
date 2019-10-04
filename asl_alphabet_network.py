import warnings  # NOQA

warnings.simplefilter(action="ignore", category=FutureWarning)  # NOQA

import os
import shutil
import click
import numpy as np
import math

from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D  # Dropout
from keras.callbacks import ModelCheckpoint

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


def get_model() -> Sequential:
    """Returns the Keras model"""

    model = Sequential()

    model.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=(200, 200, 1)))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    # model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(26, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def recommended_generator_params(data_dir, min_batch_size=32, basic=True):
    """
    Tries to determine the lowest batch size that evenly divides the number of samples
    in data_dir, starting at min_batch_size.

    If basic is True, it will return math.ceil(nsamples / min_batch_size)

    Returns: (batch_size, steps_per_epoch)
    """

    length = len([f for f in os.listdir(data_dir) if not os.path.isdir(f)])

    if basic:
        return (min_batch_size, math.ceil(length / min_batch_size))

    batch_size = min_batch_size

    while length % batch_size != 0:
        batch_size += 1

    return (batch_size, length / batch_size)


def training_data_generator(data_dir, batch_size=32):
    """Provides the training_data in the form of a generator"""

    files = [f for f in os.listdir(data_dir) if not os.path.isdir(f)]
    files = [(os.path.join(data_dir, f), f[0]) for f in files]
    files = list(filter(lambda x: not os.path.isdir(x[0]), files))

    np.random.shuffle(files)

    for i in range(0, len(files), batch_size):
        x_train = []
        y_train = []

        for (path, letter) in files[i : i + batch_size]:
            image = Image.open(path)

            # Convert to grayscale
            pixels = [(r + g + b) / 3 for (r, g, b) in image.getdata()]

            # Reshape
            x_train_i = np.array(pixels).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

            y_train_i = np.zeros((26))
            y_train_i[ord(letter) - 65] = 1

            x_train.append(x_train_i)
            y_train.append(y_train_i)

        x_train = np.reshape(x_train, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        y_train = np.array(y_train)
        yield (x_train, y_train)


def model_train(data_dir):
    """Trains the model"""

    model = get_model()
    mc = ModelCheckpoint("best_model.h5", monitor="val_loss", mode="min")

    batch_size, steps_per_epoch = recommended_generator_params(
        data_dir, min_batch_size=2, basic=True
    )
    data_generator = training_data_generator(data_dir, batch_size=batch_size)

    print(f"{'-'*10}")
    print("Batch Size", batch_size)
    print("Steps Per Epoch", steps_per_epoch)
    print(f"{'-'*10}")

    # TODO: Add back validation
    model.fit_generator(
        data_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=15,
        callbacks=[mc],
        verbose=1,
    )


def model_test(x_test, y_train):
    """Tests the model."""
    pass


def extract_data(root):
    """
    Extracts into the root directory each folder containing
    test data for a particular ASL letter.
    """

    dirs = [
        os.path.join(root, p)
        for p in os.listdir(root)
        if len(p) == 1 and os.path.isdir(os.path.join(root, p))
    ]
    for d in dirs:
        print(f"Extracting {d}")
        files = os.listdir(d)

        for file in files:
            dest = os.path.join(root, file)
            shutil.copyfile(os.path.join(d, file), dest)
        shutil.rmtree(d)


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
        model_train(data_dir)

    elif test:
        test()

    elif extract:
        extract_data(data_dir)


if __name__ == "__main__":
    main()
