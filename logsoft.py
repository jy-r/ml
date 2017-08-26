import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from dta_prep import get_dta
from fce import *

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

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W = np.random.randn(D, K)
b = np.zeros(K)

train_cost = []
test_cost = []
alpha = 0.001
for i in range(10000):
    pYtrain = forwardSoft(Xtrain, W, b)
    pYtest = forwardSoft(Xtest, W, b)
    ctrain = cross_ent(Ytrain_ind, pYtrain)
    ctest = cross_ent(Ytest_ind, pYtest)
    train_cost.append(ctrain)
    train_cost.append(ctest)

    W -= alpha*Xtrain.T.dot(pYtrain - Ytrain_ind)
    b -= alpha*(pYtrain - Ytrain_ind).sum(axis=0)
    if i % 1000 == 0:
        print(i, ctrain, ctest)

print(classificate_rate2(Ytrain, predict(pYtrain)))
print(classificate_rate2(Ytest, predict(pYtest)))

legend1, = plt.plot(train_cost, label='train cost')
legend2, = plt.plot(test_cost, label='test cost')
plt.legend([legend1, legend2])
plt.show()

Ytrain_ind.shape

W1,b1,W2,b2 = gradAsc(Xtrain, Ytrain_ind, 4, 10, alpha, 10000)
Yhat, Z = forwardSig(Xtrain, W1, b1, W2, b2)
Yhat = predict(Yhat)
classificate_rate2(Ytrain, Yhat)
Yhattest, Z = forwardSig(Xtest, W1, b1, W2, b2)
Yhattest = predict(Yhattest)
classificate_rate2(Ytest, Yhattest)
