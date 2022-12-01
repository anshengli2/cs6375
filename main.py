
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logreg import LogisticRegression as CustomLogisticRegression
from logreg2 import LogitRegression as NewLogReg
import seaborn as sn
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV


def confusionMatrix(ytst, y_pred):
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for actual_value, predicted_value in zip(ytst, y_pred):
        # see if it's a true or false prediction
        if predicted_value == actual_value:  # t
            if predicted_value == 1:  # tp
                tp += 1
            else:  # tn
                tn += 1
        else:  # f
            if predicted_value == 1:  # fp
                fp += 1
            else:  # fn
                fn += 1

    our_confusion_matrix = [
        [tp, fn],
        [fp, tn]
    ]
    our_confusion_matrix = np.array(our_confusion_matrix)
    print(our_confusion_matrix)


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

# print(pd.DataFrame(X).describe().apply(lambda s: s.apply('{0:.5f}'.format)))
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
# print(pd.DataFrame(x_train).describe().apply(
#     lambda s: s.apply('{0:.5f}'.format)))

lr = CustomLogisticRegression()
lr.fit_sigmoid(x_train, y_train, learning_rate=0.1, num_iter=10000)
y_pred = lr.predict(x_test)
fig = plt.figure()
ax = fig.add_subplot(111)

df = pd.DataFrame({'x': [x for x in range(len(x_test))],
                   'y': [y_l for y_l in y_test],
                   'y_pred_prob': np.array(lr.predict_prob(x_test))})
df['y'] = df['y'].apply(lambda x: "UP" if x == 1 else "DOWN")
sn.scatterplot(data=df, x='x', y='y_pred_prob', style='y', hue='y')
print(df)
plt.legend(loc='upper right')
ax.set_title("Decision Boundary")
ax.set_xlabel('N/2')
ax.set_ylabel('Predicted Probability')
plt.axhline(.5, color='black')
plt.show()

warnings.filterwarnings('ignore')
learningRate = [0.001, 0.01, 0.1]
models = []
# for index, i in enumerate(learningRate):
#     lr = CustomLogisticRegression()
#     lr.fit_sigmoid(x_train, y_train, learning_rate=i, num_iter=10000)
#     print("Final weights", lr.weights)
#     print("Final cost", lr.loss_history[-1])
#     models.append([lr.loss_history, lr.iter_history])
#     y_pred = lr.predict(x_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy Score:", accuracy)

#     # plt.figure(figsize=(10, 6))
#     # plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
#     # plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
#     # plt.legend()
#     # x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
#     # x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
#     # xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
#     #                        np.linspace(x2_min, x2_max))
#     # grid = np.c_[xx1.ravel(), xx2.ravel()]
#     # probs = lr.predict_prob(grid).reshape(xx1.shape)
#     # plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
#     # Confusion Matrix
#     ax = plt.axes()
#     cf_matrix = metrics.confusion_matrix(y_test, y_pred)
#     print(cf_matrix)
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
# print("Accuracy Score:", accuracy)
# # measure accuracy
# plot_confusion_matrix(logisticRegr, x_test, y_test)
# plt.show()
