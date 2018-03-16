from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from keras import backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects
import sys


class CNN:

    def __init__(self, **kwargs):
        """Instantiates VGG-like Convnet from Keras [https://keras.io/getting-started/sequential-model-guide/]

        :param kwargs:
        """

        def __build__():

            get_custom_objects().update({'swish': Activation(CNN.swish)})

            model = Sequential()
            # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
            # this applies 32 convolution filters of size 3x3 each.
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='relu'))

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd)

        sys.stdout.write('<log> Building graph...\n')
        __build__()
        sys.stdout.write('</log>\n')

    def train(self):
        pass

    def evaluate(self):
        pass

    @staticmethod
    def swish(x):
        return K.sigmoid(x) * x
