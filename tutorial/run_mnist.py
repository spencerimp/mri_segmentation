"""
Build a MLP for mnist dataset.
Training and testing are seperated for better modularity.
"""
import os
import sys
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    MaxPooling2D,
)
from keras.layers.core import (
    Dense,
    Dropout,
    Activation,
    Flatten,
)
from keras.utils import np_utils
from keras.models import load_model


def get_accuracy(pred_seq, true_seq):
    assert len(pred_seq) == len(true_seq), "The length should be identical"
    return np.count_nonzero(true_seq == pred_seq) / len(true_seq)


def train_mlp(x_train, y_train, x_test, y_test, image_size,
              batch_size, n_classes, max_epoch):
    """ Train a MLP using Sequential module."""
    # convert the label into one-hot format in order to tune on crossentropy
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    # build network
    net = Sequential()
    net.add(Dense(512, input_shape=(image_size,)))
    net.add(Activation('relu'))
    net.add(Dropout(0.5))
    net.add(Dense(512))
    net.add(Dropout(0.5))
    net.add(Dense(n_classes))
    net.add(Activation('softmax'))

    net.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # train the network
    net.fit(x_train, y_train,
            batch_size=batch_size, nb_epoch=max_epoch,
            verbose=1, validation_data=(x_test, y_test))

    score = net.evaluate(x_test, y_test)
    print("\n")
    print("Test score: {}".format(score))
    return net


def train_cnn(x_train, y_train, x_test, y_test, image_shape,
              batch_size, n_classes, max_epoch):
    """Tria a 2D convolutional neural network.
    """
    n_kernels = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    net = Sequential()
    net.add(Convolution2D(n_kernels, kernel_size[0], kernel_size[1],
                          border_mode='valid',
                          input_shape=image_shape))
    net.add(Activation('relu'))
    net.add(Convolution2D(n_kernels, kernel_size[0], kernel_size[1]))
    net.add(Activation('relu'))
    net.add(MaxPooling2D(pool_size=pool_size))
    net.add(Dropout(0.25))

    net.add(Flatten())
    net.add(Dense(128))
    net.add(Activation('relu'))
    net.add(Dropout(0.5))
    net.add(Dense(n_classes))
    net.add(Activation('softmax'))

    net.compile(optimizer='adadelta',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    # train the network
    net.fit(x_train, y_train,
            batch_size=batch_size, nb_epoch=max_epoch,
            verbose=1, validation_data=(x_test, y_test))

    score = net.evaluate(x_test, y_test)
    print("\n")
    print("Test score: {}".format(score))
    return net


def predict(net, x_test, batch_size):
    prob = net.predict(x_test, batch_size=batch_size)
    return np.argmax(prob, axis=1)


def load_datasets(n_classes, flatten=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    image_size = np.prod(x_train.shape[1:])

    if flatten:
        # vectorize 2-D image to single array
        x_train = x_train.reshape(num_train, image_size)
        x_test = x_test.reshape(num_test, image_size)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize
    x_train /= 255
    x_test /= 255

    return x_train, y_train, x_test, y_test, image_size


def run_mlp():
    print("Use a Multi-Perceptron Network")
    n_classes = 10
    batch_size = 256
    max_epoch = 20
    network_path = './mnist_mlp.h5'
    # load datasets
    x_train, y_train, x_test, y_test, image_size = load_datasets(n_classes, True)

    # retrain the network if not exists
    if not os.path.exists(network_path):
        net = train_mlp(x_train, y_train, x_test, y_test, image_size,
                        batch_size, n_classes, max_epoch)
        net.save(network_path)
    else:
        net = load_model(network_path)
    # apply network to test images
    y_test_pred = predict(net, x_test, batch_size)
    print("Accuracy = {}%".format(100 * get_accuracy(y_test_pred, y_test)))


def run_cnn():
    print("Use Convolutional neural network.")
    n_classes = 10
    batch_size = 128
    max_epoch = 20
    network_path = './mnist_cnn.h5'
    # load datasets
    x_train, y_train, x_test, y_test, image_size = load_datasets(n_classes)
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    # image_shape = x_train.shape[1:]  # (28, 28)
    image_shape = (1, 28, 28)  # (28, 28)
    # retrain the network if not exists
    if not os.path.exists(network_path):
        net = train_cnn(x_train, y_train, x_test, y_test, image_shape,
                        batch_size, n_classes, max_epoch)
        net.save(network_path)
    else:
        net = load_model(network_path)
    # apply network to test images
    y_test_pred = predict(net, x_test, batch_size)
    print("Accuracy = {}%".format(100 * get_accuracy(y_test_pred, y_test)))


if __name__ == '__main__':
    # run_mlp()
    run_cnn()
