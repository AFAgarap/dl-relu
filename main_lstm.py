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

"""Implements LSTM-RNN class for Sentiment Analysis"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import argparse
from models.lstm import LSTM
from utils import load_data
from utils import train_test_split

BATCH_SIZE = 512
CELL_SIZE = 256
DROPOUT = 0.85
EMBED_SIZE = 300
LEARNING_RATE = 1e-2
NUM_CLASSES = 2
NUM_LAYERS = 2
SEQUENCE_LENGTH = 200


def parse_args():
    parser = argparse.ArgumentParser(description='LSTM using ReLU for Sentiment Analysis')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-x', '--dataset_features', required=True, type=str,
                       help='the path where the movie reviews are found')
    group.add_argument('-y', '--dataset_labels', required=True, type=str,
                       help='the path where the review labels are found')
    group.add_argument('-s', '--checkpoint_path', required=True, type=str,
                       help='the path where to save the trained model')
    group.add_argument('-l', '--log_path', required=True, type=str,
                       help='the path where to save the TensorBoard logs')
    group.add_argument('-b', '--batch_size', required=True, type=int,
                       help='the number of data to be used by batch')
    group.add_argument('-c', '--cell_size', required=True, type=int,
                       help='the number of LSTM cells per RNN state')
    group.add_argument('-d', '--dropout_rate', required=True, type=float,
                       help='the dropout probability for the neural network')
    group.add_argument('-e', '--embed_size', required=True, type=int,
                       help='the size of the embedding layer')
    group.add_argument('-a', '--learning_rate', required=True, type=float,
                       help='the amount how fast a neural network is supposed to learn')
    group.add_argument('-n', '--num_layers', required=True, type=int,
                       help='the number of RNN layers')
    group.add_argument('-g', '--epochs', required=True, type=int,
                       help='the number of passes through the entire dataset')
    arguments = parser.parse_args()
    return arguments


def main(argv):

    features_path = argv.dataset_features
    labels_path = argv.dataset_labels

    reviews, labels, vocab_to_int = load_data(features=features_path, labels=labels_path,
                                              sequence_length=SEQUENCE_LENGTH)

    train_dataset, validation_dataset, test_dataset, num_words = train_test_split(reviews, labels, train_fraction=0.8,
                                                                                  validation_fraction=0.5,
                                                                                  vocabulary_to_integer=vocab_to_int,
                                                                                  batch_size=argv.batch_size)

    model = LSTM(alpha=argv.learning_rate, batch_size=argv.batch_size, cell_size=argv.cell_size,
                 embed_size=argv.embed_size, num_classes=NUM_CLASSES, num_layers=argv.num_layers,
                 sequence_length=SEQUENCE_LENGTH, num_words=num_words)
    model.train(epochs=argv.epochs, log_path=argv.log_path, checkpoint_path=argv.checkpoint_path,
                dropout_rate=argv.dropout_rate, train_features=train_dataset[0], train_labels=train_dataset[1],
                validation_features=validation_dataset[0], validation_labels=validation_dataset[1])
    model.predict(test_features=test_dataset[0], test_labels=test_dataset[1], checkpoint_path=argv.checkpoint_path)


if __name__ == '__main__':
    args = parse_args()

    main(argv=args)
