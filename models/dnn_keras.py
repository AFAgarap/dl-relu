# Copyright 2018 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DNN-ReLU class written using Keras/TensorFlow"""
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import sys


class DNN:

    def __init__(self, **kwargs):
        """Instantiates DNN-ReLU class

        :param kwargs:
        """
        assert 'activation' in kwargs, 'KeyNotFound : {}'.format('activation')
        assert type(kwargs['activation']) is str, \
            'Expected data type : str, but {} is {}'.format(kwargs['activation'], type(kwargs['activation']))

        assert 'batch_size' in kwargs, 'KeyNotFound : {}'.format('batch_size')
        assert type(kwargs['batch_size']) is int,\
            'Expected data type : int, but {} is {}'.format(kwargs['batch_size'], type(kwargs['batch_size']))

        assert 'dropout_rate' in kwargs, 'KeyNotFound : {}'.format('dropout_rate')
        assert type(kwargs['dropout_rate']) is float, \
            'Expected data type : float, but {} is {}'.format(kwargs['dropout_rate'], type(kwargs['dropout_rate']))

        assert 'num_classes' in kwargs, 'KeyNotFound : {}'.format('num_classes')
        assert type(kwargs['num_features']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['num_classes'], type(kwargs['num_classes']))

        assert 'num_features' in kwargs, 'KeyNotFound : {}'.format('num_features')
        assert type(kwargs['num_features']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['num_features'], type(kwargs['num_features']))

        assert 'num_neurons' in kwargs, 'KeyNotFound : {}'.format('num_neurons')
        assert type(kwargs['num_neurons']) is int, \
            'Expected data type : int, but {} is {}'.format(kwargs['num_neurons'], type(kwargs['num_neurons']))

        def __graph__():
            model = Sequential()
            model.add(Dense(kwargs['num_neurons'], activation=kwargs['activation'], input_dim=kwargs['num_features']))
            model.add(Dropout(kwargs['dropout_rate']))
            model.compile(loss=kwargs['loss'], optimizer=kwargs['optimizer'], metrics=['accuracy'])

        sys.stdout.write('<log> Building graph...\n')
        __graph__()
        sys.stdout.write('</log>\n')
