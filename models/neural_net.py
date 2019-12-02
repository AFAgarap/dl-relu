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

__version__ = "1.0.0"
__author__ = "Abien Fred Agarap"

import tensorflow as tf


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NeuralNet, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(
            units=kwargs["units"][0],
            activation=kwargs["activation"],
            kernel_initializer=kwargs["initializer"],
        )
        self.hidden_layer_2 = tf.keras.layers.Dense(
            units=kwargs["units"][1],
            activation=kwargs["activation"],
            kernel_initializer=kwargs["initializer"],
        )
        self.output_layer = tf.keras.layers.Dense(units=kwargs["num_classes"])
        self.optimizer = tf.optimizers.SGD(learning_rate=3e-4, momentum=9e-1)

    @tf.function
    def call(self, features):
        activation = self.hidden_layer_1(features)
        activation = self.hidden_layer_2(activation)
        output = self.output_layer(activation)
        return output
