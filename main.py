# -*- coding: utf-8 -*-
#Imoprting Library
import numpy as np
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('dt.csv')
dataset = dataset.drop('id',axis=1)
dataset = dataset.drop('age',axis=1)
dataset = dataset.drop('bgr',axis=1)
dataset = dataset.drop('bu',axis=1)
dataset = dataset.drop('wbcc',axis=1)

X_random = dataset.iloc[:, :-1].values
y_random = dataset.iloc[:, -1].values

#Non Categorial Missing Data handling
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='median',axis=0)

imputer = imputer.fit(X_random[:,0:10])
X_random[:,0:10] = imputer.transform(X_random[:,0:10])

#Categorial Missing Data Handle
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
y_random = labelEncoder.fit_transform(y_random)
X_random[:,10] = labelEncoder.fit_transform(X_random[:,10])
X_random[:,11] = labelEncoder.fit_transform(X_random[:,11])
X_random[:,12] = labelEncoder.fit_transform(X_random[:,12])
X_random[:,13] = labelEncoder.fit_transform(X_random[:,13])
X_random[:,14] = labelEncoder.fit_transform(X_random[:,14])
X_random[:,15] = labelEncoder.fit_transform(X_random[:,15])
X_random[:,16] = labelEncoder.fit_transform(X_random[:,16])
X_random[:,17] = labelEncoder.fit_transform(X_random[:,17])
X_random[:,18] = labelEncoder.fit_transform(X_random[:,18])
X_random[:,19] = labelEncoder.fit_transform(X_random[:,19])


#Cross Validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=12,shuffle=True,random_state=0)
for train_index,test_index in kfold.split(X_random):
    X_random_train, X_random_test = X_random[train_index], X_random[test_index]
    X_nural_train, X_nural_test = X_random[train_index], X_random[test_index]

for train_index,test_index in kfold.split(y_random):
    y_random_train, y_random_test = y_random[train_index], y_random[test_index]
    y_nural_train, y_nural_test = y_random[train_index], y_random[test_index]

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=800,random_state=0)
regressor.fit(X_random_train,y_random_train)
regressor.fit(X_random_test,y_random_test)

#predic data
random_train_pred = regressor.predict(X_random_train)

random_test_pred = regressor.predict(X_random_test)
print(random_test_pred)


from sklearn.metrics import accuracy_score
print("Random Forest Regressor : ")
print("Train data Accuracy :",accuracy_score(y_random_train,random_train_pred.round())*100)
print("Test data Accuracy :",accuracy_score(y_random_test,random_test_pred.round())*100)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_random_train,random_train_pred.round()))
print(classification_report(y_random_train,random_train_pred.round()))



#from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(20,10,30), activation='relu', solver='adam', max_iter=800)
mlp.fit(X_nural_train,y_nural_train)
mlp.fit(X_nural_test,y_nural_test)

nural_train_pred = mlp.predict(X_nural_train)
nural_test_pred = mlp.predict(X_nural_test)
print(nural_test_pred)
print("Artifical Nural Network : ")
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_nural_train,nural_train_pred))
print(classification_report(y_nural_train,nural_train_pred))

from sklearn.metrics import accuracy_score
print("Train Data Accuracy:",accuracy_score(y_nural_train, nural_train_pred)*100)
print("Test Data Accuracy:",accuracy_score(y_nural_test, nural_test_pred)*100)

print("Comparision Between Two Predicted Output is given Below")
print(len(random_test_pred))
print(len(nural_test_pred))

for i in range(len(random_test_pred)):
    s=int(random_test_pred[i].round())
    ss=nural_test_pred[i]
    if s==0 and ss==0:
        print("low")
    elif s==0 and ss==1:
        print("medium")
    elif s==1 and ss==0:
        print("medium")
    else:
        print("high")

