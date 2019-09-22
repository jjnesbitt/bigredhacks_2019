import warnings  # NOQA

warnings.simplefilter(action="ignore", category=FutureWarning)  # NOQA

import os
import shutil
import click
import numpy as np
import time

from PIL import Image, ImageFilter
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


def get_model() -> Sequential:
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(200, 200, 1)))
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(26, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model



def model_train(
    data_dir, batch_size=None, epochs=1, validation_split=0.0):
    #NAME = "model-v0-{}".format(int(time.time()))
    #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    files = [f for f in os.listdir(data_dir) if not os.path.isdir(f)]
    np.random.shuffle(files)

    #files = files[:int(len(files)/100)]
    model = get_model()
    x_train = []
    y_train = []
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min')

    for i, f in enumerate(files):
        if i % int(len(files)/100) == 0:
            print(f"{100*i / int(len(files))}%")

        path = os.path.join(data_dir, f)
        if os.path.isdir(path):
            continue
        image = Image.open(path).filter(ImageFilter.FIND_EDGES)

        # Convert to grayscale
        pixels = [(r + g + b) / 3 for (r, g, b) in image.getdata()]

        # Reshape
<<<<<<< HEAD
        x_train_i = np.array(pixels).reshape(IMAGE_WIDTH, IMAGE_HEIGHT)

        y_train_i = np.zeros((26))
        y_train_i[ord(f[0]) - 65] = 1

        x_train.append(x_train_i)
        y_train.append(y_train_i)

<<<<<<< HEAD
    x_train = np.reshape(x_train, (-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1))
    y_train = np.array(y_train)
    model.fit(x_train, y_train, epochs = 10, callbacks = [mc], validation_split = 0.3, verbose = 1)
=======
=======
        x_train = np.array(pixels).reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)

        y_train = np.zeros((1, 26))
        y_train[0][ord(f[0]) - 65] = 1

>>>>>>> eba7f24972f5bd416539dd7d7d8838c69f7f5e53
        model.fit(
            x_train,
            y_train,
            batch_size,
            epochs,
            validation_split,
            callbacks=[tensorboard],
        )
<<<<<<< HEAD
>>>>>>> eba7f24972f5bd416539dd7d7d8838c69f7f5e53
=======
>>>>>>> eba7f24972f5bd416539dd7d7d8838c69f7f5e53


def model_test(x_test, y_train):
    pass


def extract_data(root):
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
