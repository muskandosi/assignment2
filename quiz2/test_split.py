
from sklearn import datasets
import numpy as np

import ass51


def  test_split_100():
	digits = datasets.load_digits()
	data=digits.images[:100]
	target=digits.target[:100]
	data = data.reshape((100, -1))
	X_train, X_test, y_train, y_test, X_val, y_val = ass51.create_split(data,target)
	assert len(X_train)==70
	assert len(X_test)==10
	assert len(X_val)==20
	assert len(X_train)+len(X_val)+len(X_test)==100	

def test_split_9():
	digits = datasets.load_digits()
	data=digits.images[:9]
	target=digits.target[:9]
	data = data.reshape((9, -1))
	X_train, X_test, y_train, y_test, X_val, y_val = ass51.create_split(data,target)
	assert len(X_train)==6
	assert len(X_test)==1
	assert len(X_val)==2
	assert len(X_train)+len(X_val)+len(X_test)==9	



	