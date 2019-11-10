import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x = tf.linspace(-5., 5., 1000)

def softplus(x):
    return tf.math.log(tf.add(1., tf.exp(x)))

with tf.GradientTape(True) as tape:
    tape.watch(x)
    y_logistic = tf.nn.sigmoid(x)
    y_tanh = tf.nn.tanh(x)
    y_relu = tf.nn.relu(x)
    y_lrelu = tf.nn.leaky_relu(x)
    y_softplus = softplus(x)
    y_elu = tf.nn.elu(x)
dlogistic = tape.gradient(y_logistic, x)
dtanh = tape.gradient(y_tanh, x)
drelu = tape.gradient(y_relu, x)
dlrelu = tape.gradient(y_lrelu, x)
dsoftplus = tape.gradient(y_softplus, x)
delu = tape.gradient(y_elu, x)

plt.figure(figsize=(16, 8))
plt.rcParams.update({'font.size': 14})

plt.subplot(231)
plt.plot(x, y_logistic, label='logistic')
plt.plot(x, dlogistic, '--', label='logistic - derivative')
plt.legend(loc='upper left')
plt.grid()

plt.subplot(232)
plt.plot(x, y_tanh, label='tanh')
plt.plot(x, dtanh, '--', label='tanh - derivative')
plt.legend(loc='lower right')
plt.grid()

plt.subplot(233)
plt.plot(x, y_relu, label='relu')
plt.plot(x, drelu, '--', label='relu - derivative')
plt.legend(loc='upper left')
plt.grid()

plt.subplot(234)
plt.plot(x, y_lrelu, label='leaky_relu')
plt.plot(x, dlrelu, '--', label='leaky_rellu - derivative')
plt.legend(loc='upper left')
plt.grid()

plt.subplot(235)
plt.plot(x, y_softplus, label='softplus')
plt.plot(x, dsoftplus, '--', label='softplus - derivative')
plt.legend(loc='upper left')
plt.grid()

plt.subplot(236)
plt.plot(x, y_elu, label='elu')
plt.plot(x, delu, '--', label='elu - derivative')
plt.legend(loc='upper left')
plt.grid()

plt.savefig('activations.png', dpi=300)
plt.show()
