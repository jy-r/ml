import numpy as np
import pandas as pd
# from fce import *
import matplotlib.pyplot as plt
from dta_titanic import dtget
import tensorflow as tf

dta_train, dta_test, dta, ids = dtget()
Y = dta['Survived']

dta_test2 = dta_test.as_matrix()

size_of_sample = 150

dta_train_sample = dta_train[:-size_of_sample]
Y_sample = Y[:-size_of_sample]

dta_train_test = dta_train[-size_of_sample:]
Y_test = Y[-size_of_sample:]

X = dta_train_sample.as_matrix()
X_test = dta_train_test.as_matrix()
Y = Y_sample.as_matrix()
Y_test = Y_test.as_matrix()
T = np.zeros((len(Y), 2))
T_test = np.zeros((len(Y_test), 2))


M = 15
D = X.shape[1]
K = T.shape[1]


for i in range(len(Y)):
    T[i, Y[i]] = 1

for i in range(len(Y_test)):
    T_test[i, Y_test[i]] = 1

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

for i in range(200000):
    sess.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: T})
    pred_test = sess.run(predict_op, feed_dict={tfX: X_test, tfY: T_test})
    Y_solution = sess.run(predict_op, feed_dict={tfX: dta_test2})
    if i % 1000 == 0:
        print("it:",i,"train:",np.mean(Y == pred),"test:",np.mean(Y_test == pred_test))

solution = pd.DataFrame({"PassengerId": ids, "Survived": Y_solution})
solution.to_csv(("titanic/solution_tf.csv"))
