import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fce import *
from dta_titanic import dtget

dta_train, dta_test = dtget()


X = dta_train.as_matrix()
n = X.shape[1]
m = X.shape[0]
Y = dta['Survived'].as_matrix()
Theta1 = np.zeros([n,8])
Theta2 = np.zeros([9])
