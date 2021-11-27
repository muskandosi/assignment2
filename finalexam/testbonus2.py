from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np
import bonus1
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

def test_samepred():
    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = bonus1.create_split(X,y)
    pred1=bonus1.hypertrain(X_train,y_train,X_test,y_test,100,10,'relu')
    pred2=bonus1.hypertrain(X_train,y_train,X_test,y_test,100,10,'relu')
    assert(pred1.all()==pred2.all())