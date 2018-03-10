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

"""ReLU Classifier"""
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import tensorflow as tf

class MLP:

	def __init__(self, alpha, batch_size, cell_size, **kwargs):
		self.alpha = alpha
		self.batch_size = batch_size
		self.cell_size = cell_size

		def __build__():
			with tf.name_scope('inputs'):
				x_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['feature_size']], name='features')
				y_input = tf.placeholder(dtype=tf.float32, shape=[None, kwargs['num_classes']], name='labels')

				first_layer = {'h1_weights': self.initialize_weight(name='h1_weights', shape=[kwargs['feature_size'], cell_size[0]]),
				'h1_biases': self.initialize_bias(name='h1_biases', shape=cell_size[0])}

				second_layer = {'h2_weights': self.initialize_weight(name='h2_weights', shape=[cell_size[0], cell_size[1]]),
				'h2_biases': self.initialize_bias(name='h2_biases', shape=cell_size[1])}

				first_layer_activation = tf.nn.relu(tf.matmul(x_input, first_layer['h1_weights']) + first_layer['h1_biases'])

				second_layer_logits = tf.matmul(first_layer_activation, second_layer['h2_weights']) + second_layer['h2_biases']

				second_layer_activation = tf.nn.relu(second_layer_logits)

				with tf.name_scope('metrics'):
					prediction = tf.identity(second_layer_activation, name='prediction')
					with tf.name_scope('loss'):
						loss = tf.losses.hinge_loss(labels=y_input, logits=prediction)
					with tf.name_scope('accuracy'):
						correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
						accuracy = tf.reduce_mean(correct_prediction)

				with tf.name_scope('train_operation'):
					train_step = tf.train.GradientDescentOptimizer(learning_rate=self.alpha).minimize(loss)

			self.x_input = x_input
			self.y_input = y_input
			self.logits = second_layer_logits
			self.predictions = prediction
			self.loss = loss
			self.accuracy = accuracy

		sys.stdout.write('<log>Building graph...\n')
		__build__()
		sys.stdout.write('</log>\n')

	def train(self, **kwargs):
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session() as sess:
			sess.run(init_op)

			for step in range()

	def initialize_weight(self, name, shape):
		xav_init = tf.contrib.layers.xavier_initializer
		initial_values = tf.get_variable(name=name, initializer=xav_init(), shape=shape)
		return initial_values

	def initialize_bias(self, name, shape):
		initial_values = tf.constant([0.1], shape=shape)
		return initial_values


def main():
	features = datasets.load_breast_cancer().data
	labels = datasets.load_breast_cancer().target

	features = StandardScaler().fit_transform(features)

	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, stratify=labels)

if __name__ == '__main__':
	main()