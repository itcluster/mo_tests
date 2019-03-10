import numpy
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = numpy.loadtxt('pima-indians-diabetes.data.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = xgboost.XGBClassifier()
model.fit(X_train, y_train)
