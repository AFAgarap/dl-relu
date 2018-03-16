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

    assert embeddings_path is not None, 'Path to embeddings must be provided.'
    padded_features, tokens, vocabulary_size = tokenize_text(raw_text=features)
    embedding_matrix = load_embeddings(embeddings_path, max_length, tokens, vocabulary_size)
    return padded_features, embedding_matrix, vocabulary_size
