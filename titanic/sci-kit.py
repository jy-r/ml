import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from titanic.dta_titanic import dtget

dta_train, dta_test, dta, ids = dtget()
Y = dta['Survived']


size_of_sample = 150

dta_train_sample = dta_train[:-size_of_sample]
Y_sample = Y[:-size_of_sample]

dta_train_test = dta_train[-size_of_sample:]
Y_test = Y[-size_of_sample:]

X = dta_train_sample.as_matrix()
X_test = dta_train_test.as_matrix()
Y = Y_sample.as_matrix()
Y_test = Y_test.as_matrix()
T = np.zeros((len(Y), 2))
T_test = np.zeros((len(Y_test), 2))

for i in range(len(Y_test)):
    T_test[i, Y_test[i]] = 1

for i in range(len(Y)):
    T[i, Y[i]] = 1


model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=2000)
model.fit(X,T)
train_accuracy = model.score(X,T)
test_accuracy = model.score(X_test, T_test)
print(train_accuracy, test_accuracy)
