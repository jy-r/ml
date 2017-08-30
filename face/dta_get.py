import pandas as pd
import numpy as np


def dtaprep():
    dta = pd.read_csv('face/dta/fer2013.csv')
    dta.head
    Y = dta['emotion']
    X = dta['pixels'].str.split(" ", expand=True)
    X.astype(int)
    type(X)
    type(Y)
    dta2 = pd.concat([Y, X], axis=1)
    dta2.to_csv('face/dta/face.csv')


def dtaget():
    dta3 = pd.read_csv('face/dta/face.csv')
    X = dta3.drop('emotion', axis=1)
    X.shape
    Y = dta3['emotion']
    return(X, Y)
