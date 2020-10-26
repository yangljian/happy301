import pandas as pd
from sklearn import svm
import numpy as np


class PredictedModel:
    def __init__(self):
        x_train = pd.read_csv('datas/trainX.csv').values
        y_train = pd.read_csv('datas/trainY.csv').values
        self.clf = svm.SVR()
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


if __name__ == '__main__':
    p = PredictedModel()
    a = p.predict(np.array([1.25, 0.25, 0, 3, 0]).reshape(1, -1))
    print(a)
