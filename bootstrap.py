from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from random import randrange
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logreg import LogisticRegression as CustomLogisticRegression
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

data = pd.read_csv("training_set_0.csv")
data = data.copy()
# Convert label UP=1 and DOWN=0
data['Label'] = data['Label'].apply(lambda x: 1 if x == "UP" else 0)

data['Sector'].replace(['Consumer Discretionary', 'Health Care', 'Finance', 'Technology',
                        'Industrials', 'Miscellaneous', 'Utilities', 'Telecommunications',
                        'Real Estate', 'Energy', 'Consumer Staples', 'Basic Materials'],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
test_data = pd.read_csv("test_set_0.csv")
test_data = test_data.copy()
# Convert label UP=1 and DOWN=0
test_data['Label'] = test_data['Label'].apply(lambda x: 1 if x == "UP" else 0)

test_data['Sector'].replace(['Consumer Discretionary', 'Health Care', 'Finance', 'Technology',
                             'Industrials', 'Miscellaneous', 'Utilities', 'Telecommunications',
                             'Real Estate', 'Energy', 'Consumer Staples', 'Basic Materials'],
                            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

X = data.drop(['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
y = data['Label'].copy()

x_test_data = test_data.drop(
    ['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
y_test_data = test_data['Label'].copy()
# Standardizing the feature values
sc_x = StandardScaler()
x_test_data = sc_x.fit_transform(x_test_data)
"""
Logistic Regression
"""
warnings.filterwarnings('ignore')
learningRate = [0.001, 0.01, 0.1]
models = []
final_models = []
final_weights = []
kf = KFold(n_splits=10)
boostrap = 20
for b in range(boostrap):
    print("bootstrap number:", b)
    for i in learningRate:
        k = 1
        acc = []
        models = []
        for train_index, test_index in kf.split(X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            sc_x = StandardScaler()
            x_train = sc_x.fit_transform(x_train)
            x_test = sc_x.transform(x_test)
            lr = CustomLogisticRegression()
            lr.fit_sigmoid(x_train, y_train, learning_rate=i, num_iter=10000)
            y_pred = lr.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy Score K={}: {}".format(k, accuracy))
            acc.append(accuracy)
            models.append([accuracy, x_train, y_train, i])
            k += 1

        print("LR: {} Min: {} Max: {} Avg: {}".format(
            i, np.min(acc), np.max(acc), np.average(acc)))
