# Gradient noise addition with batch norm
# Copyright (C) 2019  Abien Fred Agarap, Joshua Cruzada, Gabby Torres
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Implementation of baseline model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = "1.0.0"
__author__ = "Abien Fred Agarap"

import argparse
import tensorflow as tf
import time


tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices("GPU")[0], True
)
tf.random.set_seed(42)


class NeuralNet(tf.keras.Model):
    def __init__(self, units, activation):
        super(NeuralNet, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(units=units, activation=activation)
        self.hidden_layer_2 = tf.keras.layers.Dense(units=units, activation=activation)
        self.output_layer = tf.keras.layers.Dense(units=10)
        self.optimizer = tf.optimizers.SGD(learning_rate=3e-4, momentum=9e-1)

    @tf.function
    def call(self, batch_features):
        activations = self.hidden_layer_1(batch_features)
        activations = self.hidden_layer_2(activations)
        return self.output_layer(activations)


def swish(z):
    return z * tf.nn.sigmoid(z)


def loss_fn(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )


def train_step(model, loss, features, labels, epoch):
    with tf.GradientTape() as tape:
        logits = model(features)
        train_loss = loss(logits, labels)
    gradients = tape.gradient(train_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return train_loss, gradients


def plot_gradients(gradients, step):
    for index, gradient in enumerate(gradients):
        if len(gradient.shape) == 1:
            tf.summary.histogram("histogram/{}-bias-grad".format(index), gradient, step)
        elif len(gradient.shape) != 1:
            tf.summary.histogram(
                "histogram/{}-weights-grad".format(index), gradient, step
            )


def train(model, loss_fn, dataset, epochs=10):

    writer = tf.summary.create_file_writer("tmp/{}".format(time.asctime()))

    with writer.as_default():
        with tf.summary.record_if(True):
            step = 0
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_accuracy = []
                for batch_features, batch_labels in dataset:

                    batch_loss, train_gradients = train_step(
                        model, loss_fn, batch_features, batch_labels, epoch
                    )

                    accuracy = tf.metrics.Accuracy()
                    accuracy(
                        tf.argmax(model(batch_features), 1), tf.argmax(batch_labels, 1)
                    )

                    epoch_loss += batch_loss
                    epoch_accuracy.append(accuracy.result())
                    plot_gradients(train_gradients, step)

                    step += 1

                epoch_loss = tf.reduce_mean(epoch_loss)
                epoch_accuracy = tf.reduce_mean(epoch_accuracy)

                tf.summary.scalar("loss", epoch_loss, step=step)
                tf.summary.scalar("accuracy", epoch_accuracy, step=step)

                if epoch != 0 and (epoch + 1) % 10 == 0:
                    print(
                        "Epoch {}/{}. Loss : {}, Accuracy : {}".format(
                            epoch + 1, epochs, epoch_loss, epoch_accuracy
                        )
                    )


def parse_args():
    parser = argparse.ArgumentParser("Baseline model")
    group = parser.add_argument_group("Arguments")
    group.add_argument(
        "-b",
        "--batch_size",
        required=False,
        default=1024,
        type=int,
        help="the number of examples per mini batch, default is 1024.",
    )
    group.add_argument(
        "-e",
        "--epochs",
        required=False,
        default=100,
        type=int,
        help="the number of passes through the dataset, default is 100.",
    )
    group.add_argument(
        "-a",
        "--activation",
        required=False,
        default="logistic",
        type=str,
        help="the activation function to be used by the network, default is logistic",
    )
    group.add_argument(
        "-n",
        "--neurons",
        required=False,
        default=512,
        type=int,
        help="the number of neurons in the network, default is 512",
    )
    arguments = parser.parse_args()
    return arguments


def main(arguments):
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    neurons = arguments.neurons
    activation = arguments.activation

    activation_list = ["logistic", "tanh", "relu", "leaky_relu", "swish"]
    assert (
        activation in activation_list
    ), "Expected [activation] is in [logistic, tanh, relu, leaky_relu, swish]"

    if activation == "leaky_relu":
        activation = tf.nn.leaky_relu
    else:
        activation = swish

    (
        (train_features, train_labels),
        (test_features, test_labels),
    ) = tf.keras.datasets.mnist.load_data()
    train_features = train_features.reshape(-1, 784) / 255.0
    train_features += tf.random.normal(
        stddev=5e-2, mean=0.0, shape=train_features.shape
    )
    test_features = test_features.reshape(-1, 784) / 255.0

    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    train_dataset = train_dataset.prefetch(batch_size * 2)
    train_dataset = train_dataset.shuffle(batch_size * 2)
    train_dataset = train_dataset.batch(batch_size, True)

    model = NeuralNet(units=neurons, activation=activation)
    start_time = time.time()
    train(model, loss_fn, train_dataset, epochs=epochs)
    print("training time : {}".format(time.time() - start_time))

    accuracy = tf.metrics.Accuracy()
    accuracy(tf.argmax(model(test_features), 1), tf.argmax(test_labels, 1))
    print("test accuracy : {}".format(accuracy.result()))


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
