import numpy as np
import cv2
import h5py
import time
import sys
from keras.models import load_model
import time

# current version
# Takes a new_width x new_height crop of the frame from the center
def crop(frame, new_height, new_width):
    nh = new_height
    nw = new_width
    cur_height, cur_width = frame.shape
    mdpt_height = int(cur_height / 2)
    mdpt_width = int(cur_width / 2)

    frame = frame[
        int(mdpt_height - nh / 2) : int(mdpt_height + nh / 2),
        int(mdpt_width - nw / 2) : int(mdpt_width + nw / 2),
    ]
    return frame


def continuous_classify(capture, new_model, new_height, new_width):
    while True:
        success, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = crop(frame, new_height, new_width)
        cv2.imshow("video1", frame)
        prediction = model.predict_classes(
            np.reshape(frame, (-1, new_height, new_width, 1))
        )[0]
        print(chr(prediction + 65))
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


def setup(fps, model_dir):
    # First setup the capture.
    # Register it and set the fps
    capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FPS, fps)

    # Next read in the model
    model = load_model(model_dir)
    return capture, model


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "ERROR: requires 4 arguments (fps, expected_height, expected_width, model_dir)"
        )
    fps, expected_height, expected_width = [int(el) for el in sys.argv[1:4]]
    model_dir = sys.argv[4]

    capture, model = setup(fps, model_dir)
    continuous_classify(capture, model, expected_height, expected_width)

    # model_dir = sys.argv[1]
    # model = load_model(model_dir)
    # data_dir = "/home/tom/"
    # files = [f for f in os.listdir(data_dir) if not os.path.isdir(f)]

