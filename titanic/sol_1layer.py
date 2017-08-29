import numpy as np
import pandas as pd
# from fce import *
import matplotlib.pyplot as plt
from dta_titanic import dtget

dta_train, dta_test, dta, ids = dtget()
Y = dta['Survived']


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


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return(x)


def softmax(x):
    x = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    return(x)


def forwardSig(X, W1, b1, W2, b2):
    Z = sigmoid(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    Y = softmax(A)
    return Y, Z


def derivateW2(Z, T, Y):
    return(Z.T.dot(T - Y))


def derivateb2(T, Y):
    return (T - Y).sum(axis=0)


def derivateW1(X, Z, T, Y, W2):
    return X.T.dot((T - Y).dot(W2.T) * Z * (1 - Z))


def derivateb1(T, Y, W2, Z):
    return((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def predict(pYX):
    return np.argmax(pYX, axis=1)


def cost(T, Y):
    tot = T * np.log(Y)
    return(tot.sum())


def classificate_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


for i in range(len(Y)):
    T[i, Y[i]] = 1


costs = []
D = X.shape[1]
M = 7
K = 2
alpha = 10e-7
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

for epoch in range(10000000):
    output, hidden = forwardSig(X, W1, b1, W2, b2)
    if epoch % 1000 == 0:
        c = cost(T, output)
        P = np.argmax(output, axis=1)
        r = classificate_rate(predict(T), P)
        f = classificate_rate(Y_test, np.argmax(forwardSig(X_test, W1, b1, W2, b2)[0], axis=1))
        print("it:", epoch, "cost:", c, "c_train:", r, "c_test:", f)
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


dta_test = dta_test.as_matrix()
Y_solution = np.argmax(forwardSig(dta_test, W1, b1, W2, b2)[0], axis=1)
solution = pd.DataFrame({"PassengerId": ids, "Survived": Y_solution})
solution.to_csv("titanic/solution.csv")
