from sklearn import datasets
import numpy as np
from joblib import dump, load
from sklearn import datasets, svm, metrics


def test_acc0():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==0)
	print(l)
	clf=load(model1svm.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

