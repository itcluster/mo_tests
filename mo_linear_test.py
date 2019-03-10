import numpy as np
import matplotlib.pyplot as plt


# For represention russian language in python))
plt.rc('font',family='Verdana')

# load data ( over 20 features! )
from sklearn.datasets import load_boston
boston = load_boston()
print("First line massive of data : \n{}".format(boston['data'][:1]))
print("____________________________________________________________________________________")

# split data on train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston['data'],boston['target'],random_state=0)

# initialization linear model and fit her
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

# prediction on train data and test data
y_pred = lr.predict((X_train))
print("Accuracy on train set: {:.2f}".format(np.mean(y_pred==y_train)))
y_pred = lr.predict((X_test))
print("Accuracy on train set: {:.2f}".format(np.mean(y_pred==y_test)))

# prediction on train data and test data, accuracy evaluation how R^2
y_pred = lr.predict((X_train))
print("R^2 Accuracy on train set: {:.2f}".format(lr.score(X_train, y_train)))
y_pred = lr.predict((X_test))
print("R^2 Accuracy on test set: {:.2f}".format(lr.score(X_test, y_test)))