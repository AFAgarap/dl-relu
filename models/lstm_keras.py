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

"""LSTM-ReLU class written using Keras/TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
import sys


class LstmRNN:

    def __init__(self, **kwargs):
        """Instantiates LSTM-RNN class

        :param kwargs:
        """

        def __build__():
            model = Sequential()
            e = Embedding(kwargs['vocabulary_size'], kwargs['max_length'], weights=[kwargs['embedding_matrix']],
                          input_length=kwargs['max_length'], trainable=kwargs['trainable'])
            model.add(e)
            model.add(LSTM(kwargs['num_neurons'][0], return_sequences=True))
            model.add(Dropout(kwargs['dropout_rate']))

            for num_neurons in kwargs['num_neurons'][1:]:
                model.add(LSTM(num_neurons, return_sequences=True))
                model.add(Dropout(kwargs['dropout_rate']))

            model.add(Dense(kwargs['num_classes'], activation='relu'))
            model.compile(loss=kwargs['loss'], optimizer=kwargs['optimizer'], metrics=['accuracy'])

            self.model = model

        sys.stdout.write('<log>Building graph...\n')
        __build__()
        sys.stdout.write('</log>\n')
