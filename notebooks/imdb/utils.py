# Copyright 2019 Abien Fred Agarap
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
"""Implementation of utility functions for text classification"""
import numpy as np
import tensorflow as tf

__author__ = "Abien Fred Agarap"
__version__ = "1.0.0"


def tokenize_text(text, max_length=50):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(text)

    vocabulary_size = len(tokenizer.word_index) + 1

    encoded_text = tokenizer.texts_to_sequences(text)
    padded_text = tf.keras.preprocessing.sequence.pad_sequences(
        encoded_text, maxlen=max_length, padding="post"
    )

    pad_id, start_id, oov_id, index_offset = 0, 1, 2, 2

    word_inverted_index = {
        value + index_offset: key for key, value in tokenizer.word_index.items()
    }
    vocabulary = {value: key for key, value in word_inverted_index.items()}

    return padded_text, vocabulary, vocabulary_size, tokenizer


def remove_stopwords(text):
    filtered_text = []
    stop_words = set(stopwords.words("english"))
    for token in text.split():
        if token not in stop_words:
            filtered_text.append(token)
    return " ".join([token for token in filtered_text])


def load_embeddings(filename, vocabulary_size, tokenizer, max_length=50):
    with open(filename, "r") as file:
        data = file.readlines()

    embeddings = {}

    for line in data:
        word, vector = line.split(" ", 1)
        embeddings[word] = np.array([float(element) for element in vector.split()])

    print("[INFO] Loaded word vectors : {}".format(len(embeddings)))

    embedding_matrix = np.zeros((vocabulary_size, max_length))

    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    print(
        "[INFO] Loaded word vectors for vocabulary with size : {}".format(
            embedding_matrix.shape[0]
        )
    )

    return embedding_matrix
