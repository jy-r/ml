import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dta_get import dta_get

dta = dta_get()
sample = 20

X = dta.loc[:, 'sepal_length':'petal_width'].as_matrix()
Y = dta.loc[:, 'y0':'y2'].as_matrix()
X_test = X[-sample:]
Y_test = Y[-sample:]
X = X[:-sample]
Y = Y[:-sample]

M = 5
D = X.shape[1]
K = Y.shape[1]


def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2


tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weight([D, M])
b1 = init_weight([M])
W2 = init_weight([M, K])
b2 = init_weight([K])

pyX = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tfY, logits=pyX))


train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(pyX, 1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    sess.run(train_op, feed_dict={tfX: X, tfY: Y})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: Y})
    if i % 10 == 0:
        print(np.mean(np.argmax(Y, axis=1) == pred))
