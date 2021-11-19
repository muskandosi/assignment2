from sklearn import datasets
import numpy as np
from joblib import dump, load
import quiz2.ass51 as modeltest


def  test_digit_correct_0():
	digits = datasets.load_digits()
	data=digits.images
	target=digits.target
	data = data.reshape((data.shape[0], -1))
	#X_train, X_test, y_train, y_test, X_val, y_val = modeltest.create_split(data,target)
	l=np.where(target==0)
	k=l[0]
	clf=load("../quiz2/model1.jooblib")
	pred=clf.predict(data[k])
	assert pred==0
