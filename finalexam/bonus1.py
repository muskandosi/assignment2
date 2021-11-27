from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.exceptions import DataConversionWarning

import sys
import os
import shutil
from joblib import dump, load
X, y = make_classification(n_samples=100, random_state=1)
def create_split(data,targets):
        X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.3, shuffle=False)
        #print(np.array(X_train).shape)
        return X_train, X_test, y_train, y_test

def hypertrain(xtrain,ytrain,xtest,ytest,iter,hidden,act):
    clf=MLPClassifier(random_state=1,hidden_layer_sizes=hidden,activation=act, max_iter=iter).fit(xtrain, ytrain)
    pred=clf.predict(xtest)
    return pred


X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = create_split(X,y)
pred=hypertrain(X_train,y_train,X_test,y_test,100,10,'relu')
acc=metrics.accuracy_score(y_test,pred)
print("Accuracy with 100 iteration and 10 hidden layers",acc)

pred=hypertrain(X_train,y_train,X_test,y_test,200,50,'relu')
acc=metrics.accuracy_score(y_test,pred)
print("Accuracy with 200 iteration and 50 hidden layers",acc)

pred=hypertrain(X_train,y_train,X_test,y_test,300,30,'relu')
acc=metrics.accuracy_score(y_test,pred)
print("Accuracy with 300 iteration and 30 hidden layers",acc)