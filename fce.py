import numpy as np
import matplotlib.pyplot as plt


# plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
# plt.show()


def relu(x):
    return x * (x > 0)


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


def forwardTanh(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    Y = softmax(A)
    return Y, Z


def forwardSoft(X, W, b):
    return softmax(X.dot(W) + b)


def normalize(x):
    return((x - x.mean()) / x.std())


def cost(T, Y):
    tot = T * np.log(Y)
    return(tot.sum())


def cross_ent(T, pY):
    return(-np.mean(T*np.log(pY)))


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


def classificate_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


def classificate_rate2(Y, P):
    return np.mean(Y == P)


def gradAsc(X, T, K, M=3, alpha=10e-5, iteration = 1000):
    costs = []
    D = X.shape[1]
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    for epoch in range(iteration):
        output, hidden = forwardSig(X, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classificate_rate(predict(T), P)
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

    return(W1,b1,W2,b2)
