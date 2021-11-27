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
import utils
from joblib import dump, load

import math

def  test_model_writing():
	digits = datasets.load_digits()
	#X,Y = digits.images,digits.target
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	X_train, X_test, y_train, y_test = train_test_split(data,digits.target, test_size=0.1, shuffle=False)
	utils.testing_function(X_train,y_train,"model1.joblib",X_test,y_test,testing=False)
	assert os.path.isfile("model1.joblib")

def test_small_data_overfit_checking():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	X_train, X_test, y_train, y_test = train_test_split(data,digits.target, test_size=0.1, shuffle=False)
	acc,f1=utils.testing_function(X_train,y_train,"model1.joblib",X_test,y_test,testing=True)
	assert acc>0.5
	assert f1>0.5


	