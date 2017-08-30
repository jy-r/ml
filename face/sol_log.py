import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dta_get import *

X, Y = dtaget()
T = np.zeros([len(Y), 7])
for i in range(len(Y)):
    T[i, Y[i]] = 1
X = X[:-1000]
Y = Y[:-1000]
T = T[:-1000]
X_test = X[-1000:]
Y_test = Y[-1000:]
T_test = T[-1000:]

T = T[:, 1]
T_test = T_test[:, 1]

M = 100
N, D = X.shape
K = 7
alpha = 5*10e-10
reg = 1

W1 = np.random.randn(D, M)/ np.sqrt(D)
b1 = np.zeros(M)
W2 = np.random.randn(M)/ np.sqrt(M)
b2 = 0


def sigmoid(x):
    return(1/(1-np.exp(-x)))


def relu(x):
    return(x * (x > 0))


def forward(X, W1, b1, W2, b2):
    Z = relu(X.dot(W1) + b1)
    Y = sigmoid(Z.dot(W2) + b2)
    return Y, Z

for i in range(1000):
    Y, Z = forward(X, W1, b1, W2, b2)
    W2 -= alpha * (Z.T.dot(Y-T) + reg*W2)
    b2 -= alpha * ((Y-T).sum() + reg*b2)
    W1 -= alpha * (X.T.dot(np.outer((Y-T), W2) * (Z > 0)) + reg*W1)
    b2 -= alpha * (np.sum(np.outer((Y-T), W2) * (Z > 0), axis=0) + reg*b1)

    if i % 100 == 0:
        Y_cross, _ = forward(X_test, W1, b1, W2, b2)
        test_accu = np.mean(T_test != Y_cross)
        cost = -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()
        print("i:",i,"accu:",test_accu, "cost:", cost)
