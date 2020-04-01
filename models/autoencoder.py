# Copyright 2018-2020 Abien Fred Agarap
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
"""TensorFlow 2.0 implementation of a vanilla autoencoder"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"

import tensorflow as tf


class AE(tf.keras.Model):
    def __init__(self, **kwargs):
        super(AE, self).__init__()
        self.encoder_layer = tf.keras.layers.Dense(
            units=kwargs["units"],
            activation=kwargs["activation"],
            kernel_initializer=kwargs["initializer"],
        )
        self.code = tf.keras.layers.Dense(
            units=kwargs["code_dim"], activation=tf.nn.sigmoid
        )
        self.decoder_layer = tf.keras.layers.Dense(
            units=kwargs["units"],
            activation=kwargs["activation"],
            kernel_initializer=kwargs["initializer"],
        )
        self.reconstructed = tf.keras.layers.Dense(
            units=kwargs["original_dim"], activation=tf.nn.sigmoid
        )

    @tf.function
    def call(self, features):
        activation = self.encoder_layer(features)
        code = self.code(activation)
        activation = self.decoder_layer(code)
        reconstructed = self.reconstructed(activation)
        return reconstructed
