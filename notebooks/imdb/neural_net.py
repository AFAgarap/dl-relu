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
"""TensorFlow 2.0 implementation of neural network with text embeddings layer"""
import tensorflow as tf

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


class NeuralNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super(NeuralNet, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=kwargs["vocabulary_size"],
            output_dim=kwargs["max_length"],
            input_length=kwargs["max_length"],
            embeddings_initializer=kwargs["embedding_initializer"],
            trainable=False,
        )
        self.flatten = tf.keras.layers.Flatten()
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
        self.output_layer = tf.keras.layers.Dense(
            units=kwargs["num_classes"], activation=tf.nn.sigmoid
        )

    @tf.function
    def call(self, features):
        embedding = self.embedding_layer(features)
        embedding = self.flatten(embedding)
        activation = self.hidden_layer_1(embedding)
        activation = self.hidden_layer_2(activation)
        output = self.output_layer(activation)
        return output
