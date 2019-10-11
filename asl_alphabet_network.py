import warnings  # NOQA

warnings.simplefilter(action="ignore", category=FutureWarning)  # NOQA

import os
import shutil
import click
import numpy as np
import time

from PIL import Image, ImageFilter
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.callbacks import TensorBoard, ModelCheckpoint

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200


def get_model() -> Sequential:
    model = Sequential()

    model.add(Conv2D(128, kernel_size=3, activation="relu", input_shape=(200, 200, 1)))
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, kernel_size=3, activation="relu")) 
    #model.add(Dropout(0.2)) 
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, kernel_size=3, activation="relu"))  
    #model.add(Dropout(0.2)) 
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(32, kernel_size=3, activation="relu"))
    #model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(26, activation="softmax"))


    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model



def model_train(data_dir):
    #NAME = "model-v0-{}".format(int(time.time()))
    #tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    files = [f for f in os.listdir(data_dir) if not os.path.isdir(f)]
    np.random.shuffle(files)

    # subset files
    files = files[:int(len(files)/5)]
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
        image = Image.open(path)

        # Convert to grayscale
        pixels = [(r + g + b) / 3 for (r, g, b) in image.getdata()]


        # Reshape
        x_train_i = np.array(pixels).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

        y_train_i = np.zeros((26))
        y_train_i[ord(f[0]) - 65] = 1

        x_train.append(x_train_i)
        y_train.append(y_train_i)

    x_train = np.reshape(x_train, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    y_train = np.array(y_train)
    model.fit(x_train, y_train, epochs = 15, callbacks = [mc], validation_split = 0.2, verbose = 1)


def model_test(model_dir, data_dir):
    model = load_model(model_dir)
    files = [f for f in os.listdir(data_dir) if not os.path.isdir(f)]
    num_correct = 0
    missed = []

    for i, f in enumerate(files):
        path = os.path.join(data_dir, f)
        image = Image.open(path)

        # Convert to grayscale
        pixels = [(r + g + b) / 3 for (r, g, b) in image.getdata()]
        
        # Reshape
        x_test_i = np.array(pixels).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

        # Now test
        prediction = model.predict_classes(np.reshape(x_test_i, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)))[0]
        predicted_letter = chr(prediction + 65)
        correct_letter = f[0]
        if predicted_letter == correct_letter:
            num_correct += 1
        else:
            tup = (predicted_letter, correct_letter)
            missed.append(tup)

    print("Percent Correct: {:.2f}".format(100 * num_correct / len(files)))
    for (predicted_letter, correct_letter) in missed:
        print("Incorrectly predicted a(n) {} as a(n) {}".format(correct_letter, predicted_letter))
    


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
@click.option("--model-path", help="The path to the model")
def main(train, test, extract, model, data_dir, model_path):
    if (train or test or extract) and not data_dir:
        raise click.UsageError("Data directory not specified")

    if train:
        model_train(data_dir)

    elif test:
        model_test(model_path, data_dir)

    elif extract:
        extract_data(data_dir)


if __name__ == "__main__":
    main()
