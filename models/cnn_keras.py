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
import numpy as np
from sklearn.model_selection import StratifiedKFold
import sys


class CNN:

    def __init__(self, **kwargs):
        """Instantiates VGG-like Convnet from Keras [https://keras.io/getting-started/sequential-model-guide/]

        :param kwargs:
        """

        assert 'activation' in kwargs, 'KeyNotFound : {}'.format('activation')
        assert type(kwargs['activation']) is str, \
            'Expected data type : str, but {} is {}'.format(kwargs['activation'], type(kwargs['activation']))

        assert 'input_shape' in kwargs, 'KeyNotFound : {}'.format('input_shape')
        assert type(kwargs['input_shape']) is tuple, \
            'Expected data type : tuple, but {} is {}'.format(kwargs['input_shape'], type(kwargs['input_shape']))

        assert 'loss' in kwargs, 'KeyNotFound : {}'.format('loss')
        assert type(kwargs['loss']) is str, \
            'Expected data type : str, but {} is {}'.format(kwargs['loss'], type(kwargs['loss']))

        assert 'num_classes' in kwargs, 'KeyNotFound : {}'.format('num_classes')
        assert type(kwargs['num_features']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['num_classes'], type(kwargs['num_classes']))

        assert 'optimizer' in kwargs, 'KeyNotFound : {}'.format('optimizer')
        assert type(kwargs['optimizer']) is str, \
            'Expected data type : str, but {} is {}'.format(kwargs['optimizer'], type(kwargs['optimizer']))

        def __build__():

            get_custom_objects().update({'swish': Activation(CNN.swish)})

            if kwargs['activation'] == 'swish':
                activation = CNN.swish
            else:
                activation = kwargs['activation']

            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation=activation, input_shape=kwargs['input_shape']))
            model.add(Conv2D(32, (3, 3), activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), activation=activation))
            model.add(Conv2D(64, (3, 3), activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(256, activation=activation))
            model.add(Dropout(0.5))
            model.add(Dense(kwargs['num_classes'], activation='relu'))

            if kwargs['optimizer'] == 'sgd':
                optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            else:
                optimizer = kwargs['optimizer']

            model.compile(loss=kwargs['loss'], optimizer=optimizer)

        sys.stdout.write('<log> Building graph...\n')
        __build__()
        sys.stdout.write('</log>\n')

    def train(self, **kwargs):
        """Trains the instantiated DNN-ReLU class

        :param kwargs:
        :return:
        """

        assert 'batch_size' in kwargs, 'KeyNotFound : {}'.format('batch_size')
        assert type(kwargs['batch_size']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['batch_size'], type(kwargs['batch_size']))

        assert 'n_splits' in kwargs, 'KeyNotFound : {}'.format('n_splits')
        assert type(kwargs['n_splits']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['n_splits'], type(kwargs['n_splits']))

        assert 'validation_split' in kwargs, 'KeyNotFound : {}'.format('validation_split')
        assert type(kwargs['validation_split']) is float, \
            'Expected data type : float, but {} is {}'.format(kwargs['validation_split'],
                                                              type(kwargs['validation_split']))

        assert 'verbose' in kwargs, 'KeyNotFound : {}'.format('verbose')
        assert 0 <= kwargs['verbose'] <= 2, 'ValueError : {} must be >= 0 and <= 2'.format('verbose')
        assert type(kwargs['verbose']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['verbose'], type(kwargs['verbose']))

        seed = 10
        kfold = StratifiedKFold(n_splits=kwargs['n_splits'], shuffle=True, random_state=seed)
        cvscores = []

        train_features, train_labels = kwargs['train_features'], kwargs['train_labels']

        for train, validate in kfold.split(train_features, np.argmax(train_labels, 1)):
            self.model.fit(train_features[train], train_labels[train],
                           epochs=kwargs['epochs'], batch_size=kwargs['batch_size'], verbose=kwargs['verbose'],
                           validation_split=kwargs['validation_split'])
            score = self.model.evaluate(train_features[validate], train_labels[validate],
                                        verbose=kwargs['verbose'])
            print('{} : {}, {} : {}'.format(self.model.metrics_names[0], score[0],
                                            self.model.metrics_names[1], score[1]))
            cvscores.append(score[1])
        print('==========')
        print('CV acc : {}, CV stddev : +/- {}'.format(np.mean(cvscores), np.std(cvscores)))

    def evaluate(self):
        pass

    @staticmethod
    def swish(x):
        return K.sigmoid(x) * x
