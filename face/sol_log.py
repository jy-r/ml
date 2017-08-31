from __future__ import print_function, division
from builtins import range
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from dta_get import *
from util import getBinaryData

X, Y = dtaget()
T_org = np.zeros([len(Y), 7])
for i in range(len(Y)):
    T_org[i, Y[i]] = 1
X = X[:-1000]
Y = Y[:-1000]
T_org = T_org[:-1000]
X_test = X[-1000:]
Y_test = Y[-1000:]
T_test_org = T_org[-1000:]



M = 100
N, D = X.shape
K = 7
alpha = 5*10e-10
reg = 1


W1j = []
b1j = []
W2j = []
b2j = []
cj = []
n_class = 2

def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def forward(X, W1, b1, W2, b2):
    Z = relu(X.dot(W1) + b1)
    Y = sigmoid(Z.dot(W2) + b2)
    return Y, Z

for j in range(n_class):
    T = T_org[:, j]
    T_test = T_test_org[:, j]
    cost = []
    W1 = np.random.randn(D, M)/np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M)/np.sqrt(M)
    b2 = np.zeros(1)
    for i in range(100):
        Y, Z = forward(X, W1, b1, W2, b2)
        W2 -= alpha * (Z.T.dot(Y-T) + reg*W2)
        b2 -= alpha * ((Y-T).sum() + reg*b2)
        W1 -= alpha * (X.T.dot(np.outer((Y-T), W2) * (Z > 0)) + reg*W1)
        b1 -= alpha * (np.sum(np.outer((Y-T), W2) * (Z > 0), axis=0) + reg*b1)

        if i % 10 == 0:
            Y_cross, _ = forward(X_test, W1, b1, W2, b2)
            test_accu = np.mean(T_test != np.round(Y_cross))
            train_accu = np.mean(T != np.round(Y))
            cost.append(-(T*np.log(Y) + (1-T)*np.log(1-Y)).sum())
            print("j:",j,"i:", i,"taccu:",train_accu, "accu:", test_accu)# "cost:", cost[i])
    W1j.append(W1)
    W2j.append(W2)
    b1j.append(b1)
    b2j.append(b2)
    cj.append(cost)

for j in range(n_class):
    plt.plot(cj[j])
plt.show()


Yhat = np.zeros(T_test_org.shape)
for j in range(n_class):
    Yhat[:,j], Z = forward(X_test, W1j[j], b1j[j], W2j[j], b2j[j])
yhat = np.argmax(Yhat, axis=1)
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

yhat
while True:
    x, y = X_test, yhat
    N = len(y)
    r = np.random.choice(N)
    plt.imshow(x.iloc[r,:].reshape(48, 48), cmap='gray')
    plt.title(label_map[y[r]])
    plt.show()
    prompt = input('Quit? Enter Y:\n')
    if prompt == 'Y':
        break
