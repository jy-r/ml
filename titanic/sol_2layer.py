import numpy as np
import pandas as pd
# from fce import *
import matplotlib.pyplot as plt
from dta_titanic import dtget

# Get data
dta_train, dta_test, dta, ids= dtget()
Y = dta['Survived']

# Divide sample to train and test
size_of_sample = 150

dta_train_sample = dta_train[:-size_of_sample]
Y_sample = Y[:-size_of_sample]

dta_train_test = dta_train[-size_of_sample:]
Y_test = Y[-size_of_sample:]

X = dta_train_sample.as_matrix()
X_test = dta_train_test.as_matrix()
Y = Y_sample.as_matrix()
Y_test = Y_test.as_matrix()



D = X.shape[1]
M = 5
K = 2
alpha = 10e-9

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, M)
b2 = np.random.randn(M)
W3 = np.random.randn(M, K)
b3 = np.random.randn(K)

reg = 0.01

T = np.zeros((len(Y), 2))
for i in range(len(Y)):
    T[i, Y[i]] = 1


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return(x)


def softmax(x):
    x = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    return(x)


def forward(X, W1, b1, W2, b2, W3, b3):
    A1 = X #X.shape
    Z1 = A1.dot(W1) + b1
    A2 = sigmoid(Z1) #A2.shape
    Z2 = A2.dot(W2) + b2
    A3 = sigmoid(Z2) #A3.shape
    Z3 = A3.dot(W3) + b3
    Y = softmax(Z3) #Y.shape
    return Y, Z1, Z2, Z3


def classificate_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total





costs = []


for epoch in range(20000):
    Y, Z1, Z2, Z3 = forward(X, W1, b1, W2, b2, W3, b3)
    if epoch % 1000 == 0:
        c = (T * np.log(Y)).sum()
        P = np.argmax(Y, axis=1)
        r = classificate_rate(np.argmax(T, axis=1), P)
        f = classificate_rate(Y_test, np.argmax(forward(X_test, W1, b1, W2, b2, W3, b3)[0], axis=1))
        print("it:", epoch, "cost:", c, "c_train:", r, "c_test:", f)
        costs.append(c)
        Z3.shape
        Y.shape
        T.shape

    W3 += alpha * (Z2.T.dot(T - Y) - reg*W3)
    b3 += alpha * ((T - Y).sum(axis=0) - reg*b3)
    W2 += alpha * (Z1.T.dot((T - Y).dot(W3.T) * Z2 * (1 - Z2)) - reg*W2)
    b2 += alpha * (((T - Y).dot(W3.T) * Z2 * (1 - Z2)).sum(axis=0) - reg*b2)
    W1 += alpha * (X.T.dot(((T - Y).dot(W3.T) * Z2 * (1 - Z2)).dot(W2.T) * Z1 * (1 - Z1)) - reg*W1)
    b1 += alpha * ((((T - Y).dot(W3.T) * Z2 * (1 - Z2)).dot(W2.T) * Z1 * (1 - Z1)).sum(axis=0) - reg*b1)

plt.plot(costs)
plt.show()


dta_test = dta_test.as_matrix()
Y_solution = np.argmax(forward(dta_test, W1, b1, W2, b2)[0], axis=1)
solution = pd.DataFrame({"PassengerId": ids, "Survived": Y_solution})
solution.to_csv("titanic/solution2.csv")
