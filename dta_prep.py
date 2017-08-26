import numpy as np
import pandas as pd

from fce import *


def get_dta():
    df = pd.read_csv('edta.csv')
    dta = df.as_matrix()
    X = dta[:, :-1]
    Y = dta[:, -1]
    X[:, 1] = normalize(X[:, 1])
    X[:, 2] = normalize(X[:, 2])
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]

    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    assert(np.abs(X2[:, -4:] - Z).sum() < 10e-10)

    return X2, Y


def get_bin():
    X, Y = get_dta()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


X, Y = get_dta()

M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)
