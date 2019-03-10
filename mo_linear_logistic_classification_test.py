import numpy as np
import matplotlib.pyplot as plt


# For represention russian language in python))
plt.rc('font',family='Verdana')

# load data ( over 20 features! )
from sklearn.datasets import load_breast_cancer
breast_cancer = load_breast_cancer()
print("First line massive of data : \n{}".format(breast_cancer['data'][:1]))
print("____________________________________________________________________________________")

# split data on train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(breast_cancer['data'],breast_cancer['target'],random_state=0)

# initialization linear model and fit her, C - parameter L1-regularization or L2-...
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100, penalty = 'l1')
lr.fit(X_train,y_train)

# prediction on train data and test data
y_pred = lr.predict((X_train))
print("Accuracy on train set: {:.3f}".format(np.mean(y_pred==y_train)))
y_pred = lr.predict((X_test))
print("Accuracy on train set: {:.3f}".format(np.mean(y_pred==y_test)))

# prediction on train data and test data, accuracy evaluation how R^2
y_pred = lr.predict((X_train))
print("R^2 Accuracy on train set: {:.3f}".format(lr.score(X_train, y_train)))
y_pred = lr.predict((X_test))
print("R^2 Accuracy on test set: {:.3f}".format(lr.score(X_test, y_test)))