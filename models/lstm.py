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
# =========================================================================

"""LSTM-RNN class written using TensorFlow"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import numpy as np
import os
import sys
import tensorflow as tf


class LSTM:

    def __init__(self, alpha, batch_size, cell_size, embed_size, num_classes, num_layers, num_words, sequence_length):
        """Instantiates LSTM-RNN class

        :param alpha: The learning rate to be used by the model.
        :param batch_size: The number of data to be used by batch.
        :param cell_size: The number of LSTM cells per RNN state.
        :param embed_size: The embedding layer size.
        :param num_layers: The number of RNN layers.
        :param sequence_length: The length of features to be used.
        """
        self.alpha = alpha
        self.batch_size = batch_size
        self.cell_size = cell_size
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_words = num_words
        self.sequence_length = sequence_length

        def __build__():

            tf.reset_default_graph()

            with tf.name_scope('inputs'):
                x_input = tf.placeholder(tf.int32, [None, None], name='features')
                y_input = tf.placeholder(tf.int32, [None, None], name='labels')
                keep_prob = tf.placeholder(tf.float32, name='keep_probability')

            with tf.name_scope('embeddings'):
                embedding = tf.Variable(tf.random_uniform((self.num_words, self.embed_size), -1, 1))
                embed = tf.nn.embedding_lookup(embedding, x_input)

            with tf.name_scope('rnn_layers'):
                cell = tf.contrib.rnn.MultiRNNCell([self.lstm_cell(cell_size=self.cell_size, keep_prob=keep_prob)
                                                    for _ in range(self.num_layers)])
                initial_state = cell.zero_state(self.batch_size, tf.float32)

            with tf.name_scope('rnn_forward'):
                outputs, last_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

            with tf.name_scope('dense_layer'):
                xav_init = tf.contrib.layers.xavier_initializer
                with tf.name_scope('weights'):
                    weights = tf.get_variable(name='weights', initializer=xav_init(),
                                              shape=[self.cell_size, self.num_classes])
                    self.variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = tf.get_variable(name='biases', initializer=tf.constant([0.1], shape=[self.num_classes]))
                    self.variable_summaries(biases)
                with tf.name_scope('linear_function'):
                    last = outputs[:, -1]
                    logits = tf.matmul(last, weights) + biases
                    tf.summary.histogram('logits', logits)
                with tf.name_scope('predictions'):
                    predictions = tf.nn.relu(logits, name='predictions')
                    tf.summary.histogram('predictions', predictions)

            with tf.name_scope('metrics'):
                with tf.name_scope('loss'):
                    loss = tf.losses.mean_squared_error(y_input, predictions)
                    tf.summary.scalar('loss', loss)
                with tf.name_scope('accuracy'):
                    correct_prediction = tf.equal(tf.cast(tf.round(predictions), tf.int32), y_input)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    tf.summary.scalar('accuracy', accuracy)

            with tf.name_scope('train'):
                train_step = tf.train.AdamOptimizer(self.alpha).minimize(loss)

            merged = tf.summary.merge_all()

            self.x_input = x_input
            self.y_input = y_input
            self.keep_prob = keep_prob
            self.cell = cell
            self.initial_state = initial_state
            self.last_state = last_state
            self.predictions = predictions
            self.loss = loss
            self.accuracy = accuracy
            self.train_step = train_step
            self.merged = merged
            self.embed = embed

        sys.stdout.write('<log>Building graph...\n')
        __build__()
        sys.stdout.write('</log>\n')

    def train(self, epochs=1, **kwargs):
        """Trains the instantiated LSTM-RNN object

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
                state = sess.run(self.initial_state)

                for index, (feature_batch, label_batch) in \
                        enumerate(self.next_batch(kwargs['train_features'], kwargs['train_labels'], self.batch_size),
                                  1):

                    feed_dict = {self.x_input: feature_batch, self.y_input: label_batch[:, None],
                                 self.keep_prob: kwargs['dropout_rate'], self.initial_state: state}

                    _, summary, cost, state = sess.run([self.train_step, self.merged, self.loss, self.last_state],
                                                       feed_dict=feed_dict)

                    train_writer.add_summary(summary, iteration)

                    if iteration % 5 == 0 and iteration > 0:
                        print('Epoch : {} / {}'.format(epoch + 1, epochs),
                              'Iteration : {}'.format(iteration),
                              'Train loss : {}'.format(cost))

                    if iteration % 25 == 0:
                        validation_accuracies = []

                        # validation_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))
                        validation_state = sess.run(self.initial_state)

                        for feature_batch, label_batch in \
                                self.next_batch(kwargs['validation_features'], kwargs['validation_labels'],
                                                self.batch_size):

                            feed_dict = {self.x_input: feature_batch, self.y_input: label_batch[:, None],
                                         self.keep_prob: 1, self.initial_state: validation_state}

                            summary, batch_accuracy, validation_state = sess.run([self.merged, self.accuracy,
                                                                                  self.last_state],
                                                                                 feed_dict=feed_dict)

                            validation_accuracies.append(batch_accuracy)

                        print('Validation accuracy : {}'.format(np.mean(validation_accuracies)))

                    iteration += 1

                    test_writer.add_summary(summary, iteration)

                    saver.save(sess, os.path.join(kwargs['checkpoint_path'], 'sentiment.ckpt'))

                saver.save(sess, os.path.join(kwargs['checkpoint_path'], 'sentiment.ckpt'))

    def predict(self, **kwargs):
        """Uses the trained model to perform sentiment analysis

        :param kwargs:
        :return:
        """

        test_accuracies = []

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, os.path.join(kwargs['checkpoint_path'], 'sentiment.ckpt'))

            test_state = sess.run(self.cell.zero_state(self.batch_size, tf.float32))

            for index, (feature_batch, label_batch) in \
                    enumerate(self.next_batch(kwargs['test_features'], kwargs['test_labels'], self.batch_size), 1):
                feed_dict = {self.x_input: feature_batch, self.y_input: label_batch[:, None],
                             self.keep_prob: 1.0, self.initial_state: test_state}

                batch_accuracy, test_state = sess.run([self.accuracy, self.last_state], feed_dict=feed_dict)

                test_accuracies.append(batch_accuracy)

            print('Test accuracy : {}'.format(np.mean(test_accuracies)))

    @staticmethod
    def lstm_cell(cell_size, keep_prob):
        """Creates LSTM cell with dropout

        :param cell_size: The number of LSTM cells.
        :param keep_prob: The dropout probability.
        :return:
        """

        lstm = tf.contrib.rnn.BasicLSTMCell(cell_size, reuse=tf.get_variable_scope().reuse)

        return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

    @staticmethod
    def variable_summaries(var):
        """Writes TensorBoard logs for given variable

        :param var: The variable to be logged.
        :return:
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

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
