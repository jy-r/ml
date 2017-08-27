import numpy as np
import pandas as pd
from fce import *
import matplotlib.pyplot as plt
from titanic.dta_titanic import dtget

dta_train, dta_test, dta = dtget()
Y = dta['Survived']

dta_train_sample = dta_train[:-150]
Y_sample = Y[:-150]

dta_train_test = dta_train[-150:]
Y_test = Y[-150:]

X = dta_train_sample.as_matrix()
X_test = dta_train_test.as_matrix()
Y = Y_sample.as_matrix()
Y_test = Y_test.as_matrix()
T = np.zeros((len(Y),2))

def forwardSig(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    Y = softmax(A)
    return Y, Z



for i in range(len(Y)):
    T[i, Y[i]] = 1


costs = []
D = X.shape[1]
M = 10
K = 2
alpha = 10e-7
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

for epoch in range(100000):
    output, hidden = forwardSig(X, W1, b1, W2, b2)
    if epoch % 1000 == 0:
        c = cost(T, output)
        P = np.argmax(output, axis=1)
        r = classificate_rate(predict(T), P)
        f = classificate_rate(Y_test, np.argmax(forwardSig(X_test, W1, b1, W2, b2)[0], axis=1))
        print("cost:", c, "c_train:", r, "c_test:", f)
        costs.append(c)
    output.shape
    hidden.shape
    X.shape
    W2 += alpha * derivateW2(hidden, T, output)
    b2 += alpha * derivateb2(T, output)
    W1 += alpha * derivateW1(X, hidden, T, output, W2)
    b1 += alpha * derivateb1(T, output, W2, hidden)

plt.plot(costs)
plt.show()
