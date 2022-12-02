
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

data = pd.read_csv("training_set.csv", index_col=[0])
data = data.copy()

# Convert label UP=1 and DOWN=0
data['Label'] = data['Label'].apply(lambda x: 1 if x == "UP" else 0)

data['Sector'].replace(['Consumer Discretionary', 'Health Care', 'Finance', 'Technology',
                        'Industrials', 'Miscellaneous', 'Utilities', 'Telecommunications',
                        'Real Estate', 'Energy', 'Consumer Staples', 'Basic Materials'],
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

X = data.drop(['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
y = data['Label'].copy()

# Split a dataset into a train and test set


# x_train, x_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# Standardizing the feature values
# sc_x = StandardScaler()
# X = sc_x.fit_transform(X)
# print(X)
# x_test = sc_x.transform(x_test)

"""
Logistic Regression
"""
warnings.filterwarnings('ignore')
learningRate = [0.001, 0.01, 0.1]
models = []

kf = KFold(n_splits=10)
for i in learningRate:
    k = 1
    acc = []
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.transform(x_test)
        lr = CustomLogisticRegression()
        lr.fit_sigmoid(x_train, y_train, learning_rate=i, num_iter=10000)
        # print("Final weights", lr.weights)
        # print("Final cost", lr.loss_history[-1])
        y_pred = lr.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy Score K={}: {}".format(k, accuracy))
        acc.append(accuracy)
        k += 1
    models.append(acc)
    print("LR: {} Min: {} Max: {} Avg: {}".format(
        i, np.min(acc), np.max(acc), np.average(acc)))


# for index, i in enumerate(learningRate):
#     lr = CustomLogisticRegression()
#     lr.fit_sigmoid(x_train, y_train, learning_rate=i, num_iter=10000)
#     print("Final weights", lr.weights)
#     print("Final cost", lr.loss_history[-1])
#     models.append([lr.loss_history, lr.iter_history])
#     y_pred = lr.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy Score:", accuracy)

#     # Decision boundary
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     df = pd.DataFrame({'x': [x for x in range(len(x_test))],
#                        'y': [y_l for y_l in y_test],
#                        'y_pred_prob': np.array(lr.predict_prob(x_test))})
#     df['y'] = df['y'].apply(lambda x: "UP" if x == 1 else "DOWN")
#     sn.scatterplot(data=df, x='x', y='y_pred_prob', style='y', hue='y')
#     plt.legend(loc='upper right')
#     ax.set_title("Decision Boundary   Accuracy=" + str(accuracy))
#     ax.set_xlabel('N/2')
#     ax.set_ylabel('Predicted Probability')
#     plt.axhline(.5, color='black')
#     plt.show()

#     # Confusion Matrix
#     ax = plt.axes()
#     cf_matrix = metrics.confusion_matrix(y_test, y_pred)
#     s = sn.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
#     s.set(xlabel='Predicted Label', ylabel='True Label')
#     ax.set_title("Learning rate = " +
#                  str(learningRate[index]) + "     Accuracy = " + str(accuracy))
#     plt.show()

# for i, model in enumerate(models):
#     plt.plot(model[1], model[0], label="Learning rate = " +
#              str(learningRate[i]))


# plt.xlabel("Iterations")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()

# print("\nSklearn's logistic regression")
# logisticRegr = LogisticRegression(C=1)
# logisticRegr.fit(x_train, y_train)  # apply scaling on training data
# # make predictions on the test set
# y_pred = logisticRegr.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# lr = CustomLogisticRegression()
# lr.fit_sigmoid(x_train, y_train, learning_rate=0.1, num_iter=10000)
# scores = cross_val_score(lr, x_train, y_train, cv=10)
# print('Cross-Validation Accuracy Scores', scores)
# scores = pd.Series(scores)
# print(scores.min(), scores.mean(), scores.max())
# print("Accuracy Score:", accuracy)
# plot_confusion_matrix(logisticRegr, x_test, y_test)
# plt.show()
