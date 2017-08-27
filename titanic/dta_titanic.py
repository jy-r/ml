import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def dtget():
    train_df = pd.read_csv('titanic\\dta\\train.csv')
    test_df = pd.read_csv("titanic\\dta\\test.csv")
    dta = train_df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    dta = dta.assign(Pclass1=(dta.loc[:, 'Pclass'] == 1).astype(int))
    dta = dta.assign(Pclass2=(dta.loc[:, 'Pclass'] == 2).astype(int))
    dta = dta.assign(Pclass3=(dta.loc[:, 'Pclass'] == 3).astype(int))
    dta = dta.assign(SexM=(dta.loc[:, 'Sex'] == "male").astype(int))
    dta = dta.assign(SexF=(dta.loc[:, 'Sex'] == "female").astype(int))
    dta = dta.assign(Parch0=(dta.loc[:, 'Parch'] == 0).astype(int))
    dta = dta.assign(Parch1=(dta.loc[:, 'Parch'] == 1).astype(int))
    dta = dta.assign(Parch2=(dta.loc[:, 'Parch'] == 2).astype(int))
    dta = dta.assign(Parch3=(dta.loc[:, 'Parch'] == 3).astype(int))
    dta = dta.assign(Parch4=(dta.loc[:, 'Parch'] == 4).astype(int))
    dta = dta.assign(Parch5=(dta.loc[:, 'Parch'] == 5).astype(int))
    dta = dta.assign(Parch6=(dta.loc[:, 'Parch'] == 6).astype(int))

    dta = dta.assign(EmbarkedS=(dta.loc[:, 'Embarked'] == "S").astype(int))
    dta = dta.assign(EmbarkedC=(dta.loc[:, 'Embarked'] == "C").astype(int))
    dta = dta.assign(EmbarkedQ=(dta.loc[:, 'Embarked'] == "Q").astype(int))
    dta = dta.assign(AgeN=((dta['Age']-dta['Age'].mean())/dta['Age'].std()))
    dta = dta.assign(FareN=((dta['Fare']-dta['Fare'].mean())/dta['Fare'].std()))
    dta = dta.assign(bias=1)
    dta = dta.dropna()

    dta_train = dta[['bias','Pclass1','Pclass2','SexM','AgeN','SibSp','Parch1','Parch2','Parch3','Parch4','Parch5','Parch6','FareN','EmbarkedS','EmbarkedC']]

    test_df = pd.read_csv("titanic\\dta\\test.csv")
    ids = test_df['PassengerId']
    dta_test = test_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
    dta_test = dta_test.assign(Pclass1=(dta_test.loc[:, 'Pclass'] == 1).astype(int))
    dta_test = dta_test.assign(Pclass2=(dta_test.loc[:, 'Pclass'] == 2).astype(int))
    dta_test = dta_test.assign(Pclass3=(dta_test.loc[:, 'Pclass'] == 3).astype(int))
    dta_test = dta_test.assign(SexM=(dta_test.loc[:, 'Sex'] == "male").astype(int))
    dta_test = dta_test.assign(SexF=(dta_test.loc[:, 'Sex'] == "female").astype(int))
    dta_test = dta_test.assign(Parch0=(dta_test.loc[:, 'Parch'] == 0).astype(int))
    dta_test = dta_test.assign(Parch1=(dta_test.loc[:, 'Parch'] == 1).astype(int))
    dta_test = dta_test.assign(Parch2=(dta_test.loc[:, 'Parch'] == 2).astype(int))
    dta_test = dta_test.assign(Parch3=(dta_test.loc[:, 'Parch'] == 3).astype(int))
    dta_test = dta_test.assign(Parch5=(dta_test.loc[:, 'Parch'] == 5).astype(int))
    dta_test = dta_test.assign(Parch4=(dta_test.loc[:, 'Parch'] == 4).astype(int))
    dta_test = dta_test.assign(Parch6=(dta_test.loc[:, 'Parch'] == 6).astype(int))

    dta_test = dta_test.assign(EmbarkedS=(dta_test.loc[:, 'Embarked'] == "S").astype(int))
    dta_test = dta_test.assign(EmbarkedC=(dta_test.loc[:, 'Embarked'] == "C").astype(int))
    dta_test = dta_test.assign(EmbarkedQ=(dta_test.loc[:, 'Embarked'] == "Q").astype(int))

    dta_test = dta_test.assign(AgeN =((dta_test['Age']-dta_test['Age'].mean())/dta_test['Age'].std()))
    dta_test = dta_test.assign(FareN =((dta_test['Fare']-dta_test['Fare'].mean())/dta_test['Fare'].std()))

    dta_test = dta_test.assign(bias = 1)
    dta_test = dta_test[['bias','Pclass1','Pclass2','SexM','AgeN','SibSp','Parch1','Parch2','Parch3','Parch4','Parch5','Parch6','FareN','EmbarkedS','EmbarkedC']]
    dta_test = dta_test.dropna()
    return(dta_train, dta_test, dta, ids)
