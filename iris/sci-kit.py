import numpy as np
import matplotlib.pyplot as plt
from iris.dta_get import dta_get
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

dta = dta_get()
sample = 20

X = dta.loc[:, 'sepal_length':'petal_width'].as_matrix()
Y = dta.loc[:, 'y0':'y2'].as_matrix()
X_test = X[-sample:]
Y_test = Y[-sample:]
X = X[:-sample]
Y = Y[:-sample]

model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=2000)
model.fit(X,Y)
train_accuracy = model.score(X,Y)
test_accuracy = model.score(X_test,Y_test)
print(train_accuracy, test_accuracy)
