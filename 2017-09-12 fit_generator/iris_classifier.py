from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Input

import numpy as np
import sys


def read_rows(filename, batch_size):
    while True:
        f_iris = open(filename, "r")
        for i in range(0,100):
            line = f_iris.readline()
            data = line.split(",")
            X = [float(x) for x in data[0:8]]
            X = np.reshape(X, (1, 8))

            if data[8].strip() == "setosa":
                y = 0
            elif data[8].strip() == "versicolor":
                y = 1
            elif data[8].strip() == "virginica":
                y = 2

            Y = np_utils.to_categorical(y, num_classes=3)

            if i == 0 or i % batch_size == 1:
                bufferX = X
                bufferY = Y
            else:
                bufferX = np.r_["0,2", bufferX, X]
                bufferY = np.r_["0,2", bufferY, Y]

            if i % batch_size == 0:
                yield bufferX, bufferY

        yield bufferX, bufferY
        f_iris.close()


def init_model():
    input_dimension = 8
    model = Sequential()

    model.add(Dense(10, activation="relu", input_shape=(input_dimension,)))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adadelta")

    return model


def training(argv):
    try:
        filename = argv[1]
        batch_size = int(argv[2])

        steps_per_epoch = 100 / batch_size

        model = init_model()
        model.fit_generator(read_rows(filename, batch_size), steps_per_epoch=steps_per_epoch, epochs=10, verbose=1)

    except IndexError:
        print("Usage: python iris_classifier <filename> <batch size>")
    except ValueError:
        print("Usage: python iris_classifier <filename> <batch size>")


if __name__ == '__main__':
	training(sys.argv)