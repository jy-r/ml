import pandas as pd
import numpy as np


def dta_get():
    dta = pd.read_csv('iris/dta/iris.csv')
    dta.head()
    species = pd.unique(dta['species'])

    y = 0
    for i in species:
        key = {"y{}".format(y): (dta['species'] == i).astype(int)}
        dta = dta.assign(**key)
        y += 1

    dta['sepal_length'] = (dta['sepal_length']-dta['sepal_length'].mean())/dta['sepal_length'].std()
    dta['sepal_width'] = (dta['sepal_width']-dta['sepal_width'].mean())/dta['sepal_width'].std()
    dta['petal_length'] = (dta['petal_length']-dta['petal_length'].mean())/dta['petal_length'].std()
    dta['petal_width'] = (dta['petal_width']-dta['petal_width'].mean())/dta['petal_width'].std()

    dta.drop('species', axis=1, inplace=True)
    return(dta)
