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
