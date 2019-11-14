import numpy as np
import matplotlib.pyplot as plt
from decision_stump import load_data

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

x_train, y_train = load_data('data/train.txt')
x_test, y_test = load_data('data/test.txt')

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
                         algorithm="SAMME",
                         n_estimators=5000, learning_rate=0.1)
bdt.fit(x_train, y_train)
x, y = load_data('data/test.txt')
err = np.where(bdt.predict(x) == y, 0, 1).sum() / np.size(y, 0)
print(err)
x, y = load_data('data/train.txt')
err = np.where(bdt.predict(x) == y, 0, 1).sum() / np.size(y, 0)
print(err)