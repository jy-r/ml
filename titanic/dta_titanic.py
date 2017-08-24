import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv("test.csv")
train_df.head()

dta  = dta.assign(Pclass1=(dta.loc[:, 'Pclass'] == 1).astype(int))
dta  = dta.assign(Pclass2=(dta.loc[:, 'Pclass'] == 2).astype(int))
dta  = dta.assign(Pclass3=(dta.loc[:, 'Pclass'] == 3).astype(int))

dta  = dta.assign(SexM=(dta.loc[:, 'Sex'] == "male").astype(int))
dta  = dta.assign(SexF=(dta.loc[:, 'Sex'] == "female").astype(int))

dta  = dta.assign(Parch0=(dta.loc[:, 'Parch'] == 0).astype(int))
dta  = dta.assign(Parch1=(dta.loc[:, 'Parch'] == 1).astype(int))
dta  = dta.assign(Parch2=(dta.loc[:, 'Parch'] == 2).astype(int))
dta  = dta.assign(Parch3=(dta.loc[:, 'Parch'] == 3).astype(int))
dta  = dta.assign(Parch4=(dta.loc[:, 'Parch'] == 4).astype(int))
dta  = dta.assign(Parch5=(dta.loc[:, 'Parch'] == 5).astype(int))
dta  = dta.assign(Parch6=(dta.loc[:, 'Parch'] == 6).astype(int))

dta  = dta.assign(EmbarkedS=(dta.loc[:, 'Embarked'] == "S").astype(int))
dta  = dta.assign(EmbarkedC=(dta.loc[:, 'Embarked'] == "C").astype(int))
dta  = dta.assign(EmbarkedQ=(dta.loc[:, 'Embarked'] == "Q").astype(int))

dta = dta.assign(AgeN = ((dta['Age']-dta['Age'].mean())/dta['Age'].std()))
dta = dta.assign(FareN = ((dta['Fare']-dta['Fare'].mean())/dta['Fare'].std()))

dta = dta.assign(bias = 1)
dta=dta.dropna()

dta_train  = dta[['bias','Pclass1','Pclass2','SexM','AgeN','SibSp','Parch1','Parch2','Parch3','Parch4','Parch5','Parch6','FareN','EmbarkedS','EmbarkedC']]
