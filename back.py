import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fce import *


Nclass = 500
D = 2
M = 3
K = 3
X1 = np.random.randn(Nclass, D)+np.array([0, -2])
X2 = np.random.randn(Nclass, D)+np.array([2, 2])
X3 = np.random.randn(Nclass, D)+np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)
N = len(Y)
T = np.zeros((N, K))


for i in range(N):
    T[i, Y[i]] = 1

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

alpha = 10e-7
costs = []

for epoch in range(100000):
    output, hidden = forwardSig(X, W1, b1, W2, b2)
    if epoch % 100 == 0:
        c = cost(T, output)
        P = np.argmax(output, axis=1)
        r = classificate_rate(Y, P)
        print("cost:", c, "classi:", r)
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

gradAsc(X, T, K, M=3, iteration = 1000)
