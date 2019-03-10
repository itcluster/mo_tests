import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.tree import export_graphviz

# Здесь мы инициализируем и тренируем модель дерева решений
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
clf = tree.DecisionTreeClassifier(max_depth=4, random_state=0)
clf = clf.fit(X_train, y_train)

# Здесь экспортируется дерево вроде, однако мы все данные записываем в дот_дата, как я понял
# Потом мы получем граф из этой дот_даты и записываем в пдф

import graphviz
import pydotplus
dot_data = tree.export_graphviz(clf, out_file=None)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("cancer.pdf")