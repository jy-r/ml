import numpy as np
import matplotlib.pyplot as plt

Nclass = 500

X1 = np.random.randn(Nclass, 2)+np.array([0, -2])
X2 = np.random.randn(Nclass, 2)+np.array([2, 2])
X3 = np.random.randn(Nclass, 2)+np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0]*Nclass+[1]*Nclass+[2]*Nclass)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
plt.show()


M, D = X.shape

K = 3


W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


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
    return Y


def forwardTanh(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    Y = softmax(A)
    return Y


def normalize(x):
    return((x - x.mean()) / x.std())


def classificate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def classificate_rate(Y, P):
    return np.mean(Y == P)

PYX = forward(X, W1, b1, W2, b2)
P = np.argmax(PYX, axis=1)


assert(len(P) == len(Y))
print(clasificate(Y, P))
