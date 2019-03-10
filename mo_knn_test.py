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

# initialization knn model and fit her
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

# prediction on train data and test data 
y_pred = knn.predict((X_train))
print("Accuracy on train set: {:.2f}".format(np.mean(y_pred==y_train)))
y_pred = knn.predict((X_test))
print("Accuracy on test set: {:.2f}".format(np.mean(y_pred==y_test)))