import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from dta_titanic import dtget
import pandas as pd

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

dta_test2 = dta_test.as_matrix()

accu = np.zeros([10000,1])
l = 0
act = ['identity', 'logistic', 'tanh', 'relu']
Ms = np.arange(1, 40, 1)
Ks = np.arange(1, 5, 1)
for a in act:
    for s in Ms:
        for k in Ks:
            l += 1
            hidden = np.concatenate((np.repeat(s,k),np.array([2])))
            model = MLPClassifier(hidden_layer_sizes=hidden, activation=a, max_iter=10000)
            model.fit(X,T)
            train_accuracy = model.score(X,T)
            test_accuracy = model.score(X_test, T_test)
            accu[l] = test_accuracy
            if accu[l-1] < test_accuracy:
                Y_solution = model.predict(dta_test2)
            print("act:",a,"layers:",k,"size:",s,"train:",train_accuracy,"test:", test_accuracy)

dta_test2 = dta_test.as_matrix()
Y_solution = model.predict(dta_test2)
Y_solution = np.argmax(Y_solution, axis=1)
solution = pd.DataFrame({"PassengerId": ids, "Survived": Y_solution})
solution.to_csv(("titanic/solution_sci.csv"))
