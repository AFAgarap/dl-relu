# Copyright 2018 Abien Fred Agarap

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sentiment classification using LSTM+ReLU"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

import argparse
import numpy as np
from numpy import asarray
from numpy import zeros
from keras import callbacks
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from string import punctuation

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM using ReLU for Sentiment Classification')
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
    group.add_argument('-g', '--epochs', required=True, type=int,
                       help='the number of passes through the entire dataset')
    arguments = parser.parse_args()
    return arguments


def main(argv):

    docs = '/home/darth/GitHub Projects/relu-classifier/dataset/reviews.txt'
    labels = '/home/darth/GitHub Projects/relu-classifier/dataset/labels.txt'
    docs, labels = argv.dataset_features, argv.dataset_labels    

    with open(docs, 'r') as file:
        docs = file.read()
    file.close()

    all_text = ''.join([character for character in docs if character not in punctuation])
    docs = all_text.split('\n')

    with open(labels, 'r') as file:
        labels = file.read()
    file.close()

    labels = np.array([1 if label == 'positive' else 0 for label in labels.split()])

    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    print(encoded_docs[:10])
    # [[6, 2], [3, 1], [7, 4], [8, 1], [9], [10], [5, 4], [11, 3], [5, 1], [12, 13, 2, 14]]

    # pad documents to a max length of 4 words
    max_length = 100
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs[:10])
    # [[ 6  2  0  0]
    #  [ 3  1  0  0]
    #  [ 7  4  0  0]
    #  [ 8  1  0  0]
    #  [ 9  0  0  0]
    #  [10  0  0  0]
    #  [ 5  4  0  0]
    #  [11  3  0  0]
    #  [ 5  1  0  0]
    #  [12 13  2 14]]

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('/home/darth/GitHub Projects/sequence_tagging/data/glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    # Loaded 400000 word vectors.

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    train_dataset, validation_dataset, test_dataset, _ = train_test_split(features=padded_docs, labels=labels,
        train_fraction=0.80, validation_fraction=0.50, vocabulary_to_integer=vocab_size,
        batch_size=argv.batch_size)

    train_partition = 0.80
    validation_partition = 0.50

    split_index = int(train_partition * len(padded_docs))

    train_features, validation_features = padded_docs[:split_index], padded_docs[split_index:]
    train_labels, validation_labels = labels[:split_index], labels[split_index:]

    split_index = int(validation_partition * len(validation_features))

    validation_features, test_features = validation_features[:split_index], validation_features[split_index:]
    validation_labels, test_labels = validation_labels[:split_index], validation_labels[split_index:]

    tbCallback = callbacks.TensorBoard(log_dir='./logs-2', histogram_freq=0, write_graph=True, write_images=True)

    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False)
    model.add(e)
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='relu'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(train_features, train_labels, epochs=15, validation_data=(validation_features, validation_labels), callbacks=[tbCallback])
    score, acc = model.evaluate(test_features, test_labels)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    args = parse_args()
    
    main(argv=args)
