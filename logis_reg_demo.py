
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
import pickle
import datetime

start_time = datetime.datetime.now()
# download and read mnist
mnist = fetch_mldata('MNIST original')

# 'mnist.data' is 70k x 784 array, each row represents the pixels from a 28x28=784 image
# 'mnist.target' is 70k x 1 array, each row represents the target class of the corresponding image
images = mnist.data
targets = mnist.target

# make the value of pixels from [0, 255] to [0, 1] for further process
X = mnist.data / 255.
Y = mnist.target

# split data to train and test (for faster calculation, just use 1/10 data)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X[::10], Y[::10], test_size=1000)

# TODO:use logistic regression
from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

clf = LogisticRegression()
clf.fit(X_train,Y_train)

train_accuracy = clf.score(X_train,Y_train)
test_accuracy = clf.score(X_test,Y_test)

print('Training accuracy: %0.2f%%' % (train_accuracy*100)) 
print('Testing accuracy: %0.2f%%' % (test_accuracy*100)) 

""" predictions = []
for i in range(1000):
    output = clf.predict(X_test[i].reshape(1,-1))
    predictions.append(output)

print(confusion_matrix(Y_test[0:1000],predictions))

print(classification_report(Y_test[0:1000],np.array(predictions))) """

""" train_accuracy = accuracy_score(Y_train,clf.predict(X_train))
print('Training accuracy: %0.2f%%' % (train_accuracy*100)) 

test_accuracy = accuracy_score(Y_test,predictions)
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))  """