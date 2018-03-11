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

"""Utility functions for dataset loading and preprocessing"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '0.1.0'
__author__ = 'Abien Fred Agarap'

from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
from sklearn.preprocessing import LabelEncoder
from string import punctuation


def load_embeddings(embeddings_path, max_length, tokens, vocabulary_size):
    """Returns fitted pre-trained word vectors

    :param embeddings_path: The path where the pre-trained word embeddings are found.
    :param max_length: The maximum length of word vectors.
    :param tokens: The tokenized text.
    :param vocabulary_size: The number of words in word vectors.
    """
    embeddings_index = dict()

    # load the pre-trained word vectors
    with open(embeddings_path) as file:
        data = file.readlines()

    # store <key, value> pair of pre-trained word vectors
    for line in data[1:]:
        word, vec = line.split(' ', 1)
        embeddings_index[word] = np.array([float(index) for index in vec.split()], dtype='float32')

    print('Loaded %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((vocabulary_size, max_length))
    for word, i in tokens.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_labels(labels_path, one_hot=True):
    """Returns loaded dataset labels for features

    :param labels_path: The path where the dataset labels are found.
    :param one_hot: Boolean whether to one-hot encode labels or not, default to `True`.
    """
    with open(labels_path) as file:
        labels = file.read()

    labels = [label for label in labels.split()]
    labels = LabelEncoder().fit_transform(labels)

    num_classes = np.unique(labels).shape[0]

    if one_hot:
        labels = to_categorical(labels, num_classes)
        return labels
    else:
        return labels


def clean_text(text_path):
    """Returns alphanumeric raw text

    :param text_path: The path where the dataset features (words) are found.
    """
    with open(text_path) as file:
        features = file.read()

    all_text = ''.join([character for character in features if character not in punctuation])

    features = all_text.split('\n')

    return features


def tokenize_text(raw_text, max_length=300):
    """Returns word vectors, tokenized text, and vocabulary size

    :param raw_text: The word features to be tokenized.
    :param max_length: The maximum length of word vectors.
    """
    t = Tokenizer()
    t.fit_on_texts(raw_text)
    vocabulary_size = len(t.word_index) + 1

    # integer encode the features
    encoded_features = t.texts_to_sequences(raw_text)

    # pad the sequences
    padded_features = pad_sequences(encoded_features, maxlen=max_length, padding='post')

    return padded_features, t, vocabulary_size


def vectorize_text(text_path, embeddings_path, max_length=300):
    """Returns word vectors

    :param text_path: The path where the dataset features (words) are found.
    :param embeddings_path: The path where either FastText or GloVe is found.
    :param max_length: The maximum length of word vectors, default to `300`.
    """
    features = clean_text(text_path=text_path)

    assert embeddings_path != None, 'Path to embeddings must be provided.'
    padded_features, tokens, vocabulary_size = tokenize_text(raw_text=features)
    embedding_matrix = load_embeddings(embeddings_path, max_length, tokens, vocabulary_size)
    return padded_features, embedding_matrix, vocabulary_size


def get_raw_text(reviews):
    """Returns raw text data.

    :param reviews: The loaded text data from file.
    :return:
    """
    # remove punctuation marks
    all_text = ''.join([character for character in reviews if character not in punctuation])

    # remove newline delimiters
    reviews = all_text.split('\n')

    # join all text
    all_text = ' '.join(reviews)

    # create a list of words from all_text
    words = all_text.split()

    return words, reviews


def vectorize(words, reviews):
    """Returns word vectors

    :param words: The dictionary of word counts.
    :param reviews: The reviews to be vectorized.
    :return:
    """
    counts = Counter(words)
    vocabulary = sorted(counts, key=counts.get, reverse=True)
    vocabulary_to_integer = {word: index for index, word in enumerate(vocabulary, 1)}

    reviews_integers = []
    for review in reviews:
        reviews_integers.append([vocabulary_to_integer[word] for word in review.split()])

    return reviews_integers, vocabulary_to_integer


def truncate(reviews_integers):
    """Returns truncated word vectors

    :param reviews_integers: The word vectors to be truncated
    :return:
    """
    reviews_integers = [review[0:200] for review in reviews_integers if len(review) > 0]
    return reviews_integers


def zero_pad(reviews_integers, sequence_length):
    """Returns zero left-padded features

    :param reviews_integers: The word vectors to be padded.
    :param sequence_length: The maximum length of word vectors.
    :return:
    """
    features = np.zeros((len(reviews_integers), sequence_length), dtype=int)

    for index, row in enumerate(reviews_integers):
        features[index, -len(row):] = np.array(row)[:sequence_length]

    return features


def discretize_labels(labels):
    """Returns integer-encoded labels

    :param labels: The dataset labels to be encoded to {0, 1}
    :return:
    """

    labels = np.array([1 if label == 'positive' else 0 for label in labels.split()])
    return labels


def train_test_split(features, labels, **kwargs):
    """Returns training/validation/testing dataset

    :param features: The dataset features to be split.
    :param labels: The dataset labels to be split.
    :return:
    """

    train_partition = kwargs['train_fraction']
    validation_partition = kwargs['validation_fraction']

    split_index = int(train_partition * len(features))

    train_features, validation_features = features[:split_index], features[split_index:]
    train_labels, validation_labels = labels[:split_index], labels[split_index:]

    split_index = int(validation_partition * len(validation_features))

    validation_features, test_features = validation_features[:split_index], validation_features[split_index:]
    validation_labels, test_labels = validation_labels[:split_index], validation_labels[split_index:]

    num_words = len(kwargs['vocabulary_to_integer']) + 1

    train_size = train_features.shape[0]
    val_size = validation_features.shape[0]
    test_size = test_features.shape[0]

    train_features, train_labels = train_features[:train_size - (train_size % kwargs['batch_size'])], \
                                   train_labels[:train_size - (train_size % kwargs['batch_size'])]
    validation_features, validation_labels = validation_features[:val_size - (val_size % kwargs['batch_size'])], \
                                             validation_labels[:val_size - (val_size % kwargs['batch_size'])]
    test_features, test_labels = test_features[:test_size - (test_size % kwargs['batch_size'])], \
                                 test_labels[:test_size - (test_size % kwargs['batch_size'])]

    return [train_features, train_labels], [validation_features, validation_labels], [test_features, test_labels], \
           num_words


def load_data(features, labels, **kwargs):
    """Returns preprocessed dataset

    :param features: The dataset features.
    :param labels: The dataset labels.
    :return:
    """

    with open(features, 'r') as f:
        reviews = f.read()

    with open(labels, 'r') as f:
        labels = f.read()

    words, reviews = get_raw_text(reviews)

    reviews_integers, vocabulary_to_integer = vectorize(words, reviews)

    reviews_integers = truncate(reviews_integers)

    features = zero_pad(reviews_integers, sequence_length=kwargs['sequence_length'])

    labels = discretize_labels(labels)

    return features, labels, vocabulary_to_integer
