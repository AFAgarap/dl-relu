# Copyright 2019 Abien Fred Agarap
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of feed-forward neural net"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NeuralNet, self).__init__()
        self.num_layers = kwargs['num_layers']
        self.neurons = kwargs['neurons']
        self.hidden_layers = []
        self.activation = kwargs['activation']
        for index in range(self.num_layers):
            self.hidden_layers.append(
                    tf.keras.layers.Dense(
                        units=self.neurons[index], activation=self.activation
                        )
                    )
        self.output_layer = tf.keras.layers.Dense(
                units=kwargs['num_classes'], activation=tf.nn.softmax
                )
        self.optimizer = tf.optimizers.SGD(
                learning_rate=1e-1,
                momentum=9e-1,
                decay=1e-6,
                nesterov=True
                )

    @tf.function
    def call(self, features):
        activations = []
        for index in range(self.num_layers):
            if index == 0:
                activations.append(self.hidden_layers[index](features))
            else:
                activations.append(self.hidden_layers[index](activations[index - 1]))
        output = self.output_layer(activations[-1])
        return output
