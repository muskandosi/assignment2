print("qu1")
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics

from sklearn.exceptions import DataConversionWarning

import sys
import os
import shutil
from joblib import dump, load


import math
def create_split(data,targets):
        X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.3, shuffle=False)
        X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.5, shuffle=False)
        #print(np.array(X_train).shape)
        return X_train, X_test, y_train, y_test, X_val, y_val
def testing_function(xtrain,ytrain,xval,yval,xtest,ytest,g,k):
    clf = svm.SVC(gamma=g,kernel=k)
    clf.fit(xtrain,ytrain)
    j=clf.predict(xtrain)
    acc1=metrics.accuracy_score(ytrain,j)
    acc2=metrics.accuracy_score(yval,clf.predict(xval))
    acc3=metrics.accuracy_score(ytest,clf.predict(xtest))
    return(acc1,acc2,acc3)
    

gamma=[0.0001,00.001,000.01,0000.1]
kernel=["rbf","poly","linear"]
digits = datasets.load_digits()
data=digits.images
target=digits.target
s=data.shape[0]
data = data.reshape((s, -1))
X_train, X_test, y_train, y_test, X_val, y_val = create_split(data,target)
acctrain=[]
acctest=[]
accval=[]
print("|        hyperparameters      |                    run-1                    |                    run-2                    |")
print("|    gamma    |    kernel     |     train    |      dev      |     test     |     train    |      dev      |     test     |")
for i in range(len(gamma)):

    for j in range(len(kernel)):
        print("|   ",gamma[i],end="")
        print("   |     ",kernel[j],end="")
        for k in range(2):
            acc1,acc2,acc3=testing_function(X_train,y_train,X_val,y_val,X_test,y_test,gamma[i],kernel[j])
            print("    |     ",round(acc1*100,2),end="")

            print("    |     ",round(acc2*100,2),end="")
            print("    |     ",round(acc3*100,2),end="")
            print("    | ",end="")
        print(" ")
        







