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

