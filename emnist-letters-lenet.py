from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices('GPU')[0],
        True
        )

tf.random.set_seed(42)
np.random.seed(42)

batch_size = 256
epochs = 50

data = loadmat('/home/darth/Projects/matlab/emnist-letters.mat')

train_features = data['dataset'][0][0][0][0][0][0]
test_features = data['dataset'][0][0][1][0][0][0]

train_labels = data['dataset'][0][0][0][0][0][1]
test_labels = data['dataset'][0][0][1][0][0][1]

train_features = train_features.reshape(-1, 28, 28, 1)
test_features = test_features.reshape(-1, 28, 28, 1)

train_features = train_features / 255.
test_features = test_features / 255.

train_labels = train_labels - 1
test_labels = test_labels - 1

validation_features, test_features, validation_labels, test_labels = \
        train_test_split(test_features,
                         test_labels,
                         test_size=0.50,
                         stratify=test_labels,
                         random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_features, train_labels)
        )
train_dataset = train_dataset.prefetch(batch_size * 2)
train_dataset = train_dataset.shuffle(batch_size * 2)
train_dataset = train_dataset.batch(batch_size, True)

val_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_features, validation_labels)
        )
val_dataset = val_dataset.prefetch(batch_size)
val_dataset = val_dataset.shuffle(batch_size)
val_dataset = val_dataset.batch((batch_size // 2), True)

test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_features, test_labels)
        )
test_dataset = test_dataset.prefetch(batch_size)
test_dataset = test_dataset.shuffle(batch_size)
test_dataset = test_dataset.batch((batch_size // 2), True)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        6, (5, 5), input_shape=(28, 28, 1), activation=tf.nn.relu
        ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, (5, 5), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=26, activation=tf.nn.softmax)
    ])
# model.load_weights('notebooks/saved_model/emnist-letters/lenet/1')

model.compile(loss=tf.losses.sparse_categorical_crossentropy,
              optimizer=tf.optimizers.SGD(
                  learning_rate=1e-1, momentum=9e-1, decay=1e-6
                  ),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset)

model.save_weights('saved_model/emnist-letters/lenet/1')
score = model.evaluate(test_dataset)
print('accuracy : {}'.format(score[1]))

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.legend(loc='lower right')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.ylim([min(plt.ylim()), 1])
plt.title('training and validation accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.legend(loc='upper right')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.ylim([min(plt.ylim()), 1])
plt.title('training and validation loss')

plt.show()
