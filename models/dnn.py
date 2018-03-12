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

"""DNN-ReLU class written using TensorFlow"""
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from models.loss import hinge
from models.loss import mean_squared_error
from models.loss import squared_hinge
import numpy as np
import os
import sys
import tensorflow as tf


class DNN:

    def __init__(self, alpha, batch_size, num_neurons, **kwargs):
        """Instantiates DNN-ReLU class

        :param alpha: The learning rate to be used by the neural network.
        :param batch_size: The number of data per batch to use for training/validation/testing.
        :param cell_size: The number of neurons in each hidden layer.
        :param kwargs:
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.num_neurons = num_neurons

        def __build__():
            with tf.name_scope('inputs'):
                # [BATCH_SIZE, NUM_FEATURES]
                x_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['num_features']], name='features')

                # [BATCH_SIZE, NUM_CLASSES]
                y_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['num_classes']], name='labels')

                # [NUM_FEATURES, NUM_NEURONS]
                first_layer = {'h1_weights': self.initialize_weight(name='h1_weights',
                                                                    shape=[kwargs['num_features'], num_neurons[0]]),
                               'h1_biases': self.initialize_bias(name='h1_biases', shape=num_neurons[0])}

                # [NUM_NEURONS, NUM_NEURONS]
                second_layer = {'h2_weights': self.initialize_weight(name='h2_weights',
                                                                     shape=[num_neurons[0], num_neurons[1]]),
                                'h2_biases': self.initialize_bias(name='h2_biases', shape=num_neurons[1])}

                # [NUM_NEURONS, NUM_CLASSES]
                third_layer = {'h3_weights': self.initialize_weight(name='h3_weights',
                                                                    shape=[num_neurons[1], kwargs['num_classes']]),
                               'h3_biases': self.initialize_bias(name='h3_biases', shape=kwargs['num_classes'])}

                # first hidden layer output
                first_layer_logits = tf.matmul(x_input, first_layer['h1_weights']) + first_layer['h1_biases']

                # first hidden layer non-linearity
                first_layer_activation = tf.nn.relu(first_layer_logits)

                # second hidden layer output
                second_layer_logits = tf.matmul(first_layer_activation, second_layer['h2_weights']) + \
                                      second_layer['h2_biases']

                # second hidden layer non-linearity
                second_layer_activation = tf.nn.relu(second_layer_logits)

                # fully-connected layer ouput
                third_layer_logits = tf.matmul(second_layer_activation, third_layer['h3_weights']) + \
                                     third_layer['h3_biases']

                # network prediction
                prediction = tf.nn.relu(third_layer_logits, name='prediction')

                with tf.name_scope('metrics'):
                    with tf.name_scope('loss'):
                        if kwargs['loss'] == 'hinge':
                            loss = hinge(labels=y_input, logits=prediction,
                                         num_classes=kwargs['num_classes'],
                                         penalty_parameter=kwargs['penalty_parameter'],
                                         weight=third_layer['h3_weights'])
                        elif kwargs['loss'] == 'squared_hinge':
                            loss = squared_hinge(labels=y_input, logits=prediction,
                                                 num_classes=kwargs['num_classes'],
                                                 penalty_parameter=kwargs['penalty_parameter'],
                                                 weight=third_layer['h3_weights'])
                        elif kwargs['loss'] == 'mean_squared_error':
                            loss = mean_squared_error(predicted_labels=prediction, target_labels=y_input)
                        elif kwargs['loss'] == 'binary_crossentropy':
                            loss = tf.losses.sigmoid_cross_entropy(y_input, prediction)
                        elif kwargs['loss'] == 'softmax_crossentropy':
                            loss = tf.losses.softmax_cross_entropy(y_input, prediction)
                        else:
                            loss = squared_hinge(labels=y_input, logits=prediction,
                                                 num_classes=kwargs['num_classes'],
                                                 penalty_parameter=kwargs['penalty_parameter'],
                                                 weight=third_layer['h3_weights'])
                    with tf.name_scope('accuracy'):
                        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                with tf.name_scope('train_operation'):
                    if kwargs['optimizer'] == 'sgd':
                        train_step = tf.train.GradientDescentOptimizer(learning_rate=self.alpha).minimize(loss)
                    elif kwargs['optimizer'] == 'adam':
                        train_step = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(loss)
                    elif kwargs['optimizer'] == 'adagrad':
                        train_step = tf.train.AdagradOptimizer(learning_rate=self.alpha).minimize(loss)
                    elif kwargs['optimizer'] == 'adadelta':
                        train_step = tf.train.AdadeltaOptimizer(learning_rate=self.alpha).minimize(loss)
                    elif kwargs['optimizer'] == 'rmsprop':
                        train_step = tf.train.RMSPropOptimizer(learning_rate=self.alpha).minimize(loss)
                    else:
                        train_step = tf.train.GradientDescentOptimizer(learning_rate=self.alpha).minimize(loss)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.logits = third_layer_logits
            self.predictions = prediction
            self.loss = loss
            self.accuracy = accuracy
            self.train_step = train_step
            self.merged = merged

        sys.stdout.write('<log>Building graph...\n')
        __build__()
        sys.stdout.write('</log>\n')

    def train(self, epochs=1, **kwargs):
        """Trains the instantiated GRU-RNN object

        :param epochs: The number of passes through the entire dataset.
        :param kwargs:
        :return:
        """

        initializer_op = tf.group(tf.global_variables_initializer())

        saver = tf.train.Saver()

        logs_path_train = os.path.join(kwargs['log_path'], 'training')
        log_path_test = os.path.join(kwargs['log_path'], 'testing')

        train_writer = tf.summary.FileWriter(logdir=logs_path_train, graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(logdir=log_path_test, graph=tf.get_default_graph())

        with tf.Session() as sess:

            sess.run(initializer_op)

            iteration = 1

            for epoch in range(epochs):

                for index, (feature_batch, label_batch) in \
                        enumerate(self.next_batch(kwargs['train_features'], kwargs['train_labels'], self.batch_size),
                                  1):

                    feed_dict = {self.x_input: feature_batch, self.y_input: label_batch}

                    _, cost, accuracy = sess.run([self.train_step, self.loss, self.accuracy], feed_dict=feed_dict)

                    # train_writer.add_summary(summary, iteration)

                    if iteration % 5 == 0 and iteration > 0:
                        print('Epoch : {} / {}'.format(epoch + 1, epochs),
                              'Iteration : {}'.format(iteration),
                              'Accuracy : {}'.format(accuracy),
                              'Train loss : {}'.format(cost))

                    if iteration % 25 == 0:
                        validation_accuracies = []

                        for feature_batch, label_batch in \
                                self.next_batch(kwargs['validation_features'], kwargs['validation_labels'],
                                                self.batch_size):
                            feed_dict = {self.x_input: feature_batch, self.y_input: label_batch}

                            # summary, batch_accuracy = sess.run([self.merged, self.accuracy], feed_dict=feed_dict)
                            batch_accuracy = sess.run([self.accuracy], feed_dict=feed_dict)
                            validation_accuracies.append(batch_accuracy)

                        print('Validation accuracy : {}'.format(np.mean(validation_accuracies)))

                    iteration += 1

                    # test_writer.add_summary(summary, iteration)

                    saver.save(sess, os.path.join(kwargs['checkpoint_path'], 'sentiment.ckpt'))

                saver.save(sess, os.path.join(kwargs['checkpoint_path'], 'sentiment.ckpt'))

    @staticmethod
    def initialize_weight(name, shape):
        xav_init = tf.contrib.layers.xavier_initializer
        initial_values = tf.get_variable(name=name, initializer=xav_init(), shape=shape)
        return initial_values

    @staticmethod
    def initialize_bias(name, shape):
        initial_values = tf.constant([0.1], shape=[shape], name=name)
        return initial_values

    @staticmethod
    def next_batch(features, labels, batch_size=100):
        """Returns the next batch of data

        :param features: The dataset features.
        :param labels: The dataset labels.
        :param batch_size: The number of data to return
        :return:
        """

        num_batches = len(features) // batch_size

        feature_batch, label_batch = features[:num_batches * batch_size], labels[:num_batches * batch_size]

        for index in range(0, len(features), batch_size):
            yield feature_batch[index:(index + batch_size)], label_batch[index:(index + batch_size)]