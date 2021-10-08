from skimage.transform import rescale, resize, downscale_local_mean
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
digits = datasets.load_digits()

def create_split(data,targets):
        X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.1, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2222, shuffle=False)
        #print(np.array(X_train).shape)
        return X_train, X_test, y_train, y_test, X_val, y_val


def testing_function(xtrain,ytrain,modelpath,xtest,ytest,testing=False):
	clf = svm.SVC(gamma=0.001)
	clf.fit(xtrain,ytrain)
	dump(clf, modelpath)
	if testing==True:
		ytest1=clf.predict(xtest)
		acc=metrics.accuracy_score(ytest,ytest1)
		f1=metrics.f1_score(ytest,ytest1,average='macro')
		return(acc,f1)


