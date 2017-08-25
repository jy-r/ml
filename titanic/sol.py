import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fce import *
from titanic.dta_titanic import dtget

dta_train, dta_test, dta = dtget()


X = dta_train.as_matrix()
Y = dta['Survived']

T = np.zeros((len(Y),2))

for i in range(len(Y)):
    T[i, Y[i]] = 1

T.shape

W1,b1,W2,b2 = gradAsc(X, T, 2, 15, 10e-7, 100000)
