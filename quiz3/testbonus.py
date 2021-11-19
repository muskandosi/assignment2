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

def test_acc1():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==1)
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


def test_acc2():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==2)
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


def test_acc3():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==3)
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


def test_acc4():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==4)
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


def test_acc5():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==5)
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


def test_acc6():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==6)
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


def test_acc7():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==7)
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


def test_acc8():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==8)
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


def test_acc9():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==9)
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



def test_acc9dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==9)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70



def test_acc8dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==8)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70


def test_acc7dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==7)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc6dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==6)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc5dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==5)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc4dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==4)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc3dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==3)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc2dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==2)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc1dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==1)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70

def test_acc0dt():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	target=digits.targets
	l=np.where(target==0)
	print(l)
	clf=load(modeldecisiontree.joblib)
	test=[]
	label=[]
	for i in l:
		test.append(data[i])
		label.append(target[i])
	pred=clf.predict(test)
	acc=metrics.accuracy_score(label,pred)
	assert acc>0.70