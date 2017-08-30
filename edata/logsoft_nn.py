import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from edata.dta_prep import get_dta


def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


X, Y = get_dta()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))
M = 5 # number of hidden units

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)


def softmax(x):
    x = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)
    return(x)


def forwardTanh(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    A = Z.dot(W2) + b2
    Y = softmax(A)
    return Y, Z


def classificate_rate2(Y, P):
    return np.mean(Y == P)


def predict(pYX):
    return np.argmax(pYX, axis=1)


def cross_ent(T, pY):
    return(-np.mean(T*np.log(pY)))


train_cost = []
test_cost = []
alpha = 0.001

for i in range(50000):
    pYtrain, Ztrain = forwardTanh(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forwardTanh(Xtest, W1, b1, W2, b2)

    ctrain = cross_ent(Ytrain_ind, pYtrain)
    ctest = cross_ent(Ytest_ind, pYtest)

    train_cost.append(ctrain)
    test_cost.append(ctest)

    W2 -= alpha*Ztrain.T.dot(pYtrain - Ytrain_ind)
    b2 -= alpha*(pYtrain - Ytrain_ind).sum()
    dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
    W1 -= alpha * Xtrain.T.dot(dZ)
    b1 -= alpha*dZ.sum(axis=0)

    if i % 1000 == 0:
        print(i, ctrain, ctest)

print(classificate_rate2(Ytrain, predict(pYtrain)))
print(classificate_rate2(Ytest, predict(pYtest)))

legend1, = plt.plot(train_cost, label='train cost')
legend2, = plt.plot(test_cost, label='test cost')
plt.legend([legend1, legend2])
plt.show()
