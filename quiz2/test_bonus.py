
from sklearn import datasets,metrics
import numpy as np
import os
from joblib import dump, load
import ass51


def  test_splitequal():
	digits = datasets.load_digits()
	data=digits.images[:100]
	target=digits.target[:100]
	data = data.reshape((100, -1))
	X_train, X_test, y_train, y_test, X_val, y_val = ass51.create_split(data,target)
	assert len(X_train)==len(y_train)
	assert len(X_test)==len(y_test)
	assert len(X_val)==len(y_val)


def test_dimensionality():
	digits = datasets.load_digits()
	data=digits.images[:9]
	target=digits.target[:9]
	data = data.reshape((9, -1))
	X_train, X_test, y_train, y_test, X_val, y_val = ass51.create_split(data,target)
	assert X_train[0].shape==X_val[0].shape	
	assert X_val[0].shape==X_test[0].shape	


def  test_model_not_corrupt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	X_train, X_test, y_train, y_test, X_val, y_val = ass51.create_split(data,digits.target)
	ass51.testing_function(X_train,y_train,"model1.joblib",X_test,y_test,testing=False)
	assert os.path.isfile("model1.joblib")

def test_modelsame():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	X_train, X_test, y_train, y_test, X_val, y_val = ass51.create_split(data,digits.target)
	acc,f1=ass51.testing_function(X_test,y_test,"model1.joblib",X_test,y_test,testing=True)
	acc1,f1_1=ass51.testing_function(X_test,y_test,"model1.joblib",X_test,y_test,testing=True)
	assert acc==acc1
	 
	
	
