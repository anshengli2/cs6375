
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
    # print(np.argmax(acc), models[np.argmax(acc)][2])
    final_models.append(models[np.argsort(acc)[len(acc)//2]])

models = []
for acc, x_train, y_train, learning_rate in final_models:
    lr = CustomLogisticRegression()
    lr.fit_sigmoid(x_train, y_train,
                   learning_rate=learning_rate, num_iter=10000)
    print("Learning rate: ", learning_rate)
    print("Final weights: ", lr.weights)
    print("Final cost: ", lr.loss_history[-1])
    models.append([lr.loss_history, lr.iter_history])
    y_pred = lr.predict(x_test_data)
    accuracy = accuracy_score(y_test_data, y_pred)
    print("Accuracy Score:", accuracy)

    # Decision boundary
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df = pd.DataFrame({'x': [x for x in range(len(x_test_data))],
                       'y': [y_l for y_l in y_test_data],
                       'y_pred_prob': np.array(lr.predict_prob(x_test_data))})
    df['y'] = df['y'].apply(lambda x: "UP" if x == 1 else "DOWN")
    sn.scatterplot(data=df, x='x', y='y_pred_prob', style='y', hue='y')
    plt.legend(loc='upper right')
    ax.set_title("Decision Boundary   Accuracy=" + str(accuracy))
    ax.set_xlabel('N/2')
    ax.set_ylabel('Predicted Probability')
    plt.axhline(.5, color='black')
    plt.show()

    # Confusion Matrix
    ax = plt.axes()
    cf_matrix = metrics.confusion_matrix(y_test_data, y_pred)
    s = sn.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
    s.set(xlabel='Predicted Label', ylabel='True Label')
    ax.set_title("Learning rate = " + str(learning_rate) +
                 "     Accuracy = " + str(accuracy))
    plt.show()

    ns_probs = [0 for _ in range(len(y_test_data))]
    lr_probs = np.array(lr.predict_prob(x_test_data)[0])
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test_data, ns_probs)
    lr_auc = roc_auc_score(y_test_data, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test_data, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test_data, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


for i, model in enumerate(models):
    plt.plot(model[1], model[0], label="Learning rate = " +
             str(learningRate[i]))

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()

# # Split a dataset into a train and test set
# x_train, x_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42)
# x_train = sc_x.fit_transform(x_train)
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


# print("\nSklearn's logistic regression")
# # Split a dataset into a train and test set
# x_train, x_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# logisticRegr = LogisticRegression()
# logisticRegr.fit(x_train, y_train)  # apply scaling on training data
# # make predictions on the test set
# y_pred = logisticRegr.predict(x_val)
# accuracy = accuracy_score(y_val, y_pred)
# # lr = CustomLogisticRegression()
# # lr.fit_sigmoid(x_train, y_train, learning_rate=0.1, num_iter=10000)
# scores = cross_val_score(logisticRegr, x_train, y_train, cv=10)
# print('Cross-Validation Accuracy Scores', scores)
# scores = pd.Series(scores)
# print("Min: {} Max: {}, Average: {}".format(
#     scores.min(), scores.max(), scores.mean()))
# print("Accuracy Score:", accuracy)
# plot_confusion_matrix(logisticRegr, x_val, y_val)
# plt.show()

# y_pred = logisticRegr.predict(x_test_data)
# accuracy = accuracy_score(y_test_data, y_pred)
# print("Accuracy Score:", accuracy)
# plot_confusion_matrix(logisticRegr, x_test_data, y_test_data)
# plt.show()
