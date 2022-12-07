
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
from svm import SVM
# from logreg2 import LogitRegression as NewLogReg
# import seaborn as sn
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

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardizing the feature values
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

svm = SVM()
svm.fit(x_train, y_train)
svm.predictTest(x_test, y_test)

# # support_vectors = clf.support_vectors_
# plt.scatter(y[:, 1], y[:, 0])
# # plt.plot(support_vectors[0][:], support_vectors[1][:], color='violet')
# # plt.title('Linearly separable data with support vectors')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()

# """
# Logistic Regression
# """
# warnings.filterwarnings('ignore')
# learningRate = [0.001, 0.01, 0.1]
# models = []
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

# # for i, model in enumerate(models):
# #     plt.plot(model[1], model[0], label="Learning rate = " +
# #              str(learningRate[i]))


# # plt.xlabel("Iterations")
# # plt.ylabel("Cost")
# # plt.legend()
# # plt.show()

# # print("\nSklearn's logistic regression")
# # logisticRegr = LogisticRegression(C=1)
# # logisticRegr.fit(x_train, y_train)  # apply scaling on training data
# # # make predictions on the test set
# # y_pred = logisticRegr.predict(x_test)
# # accuracy = accuracy_score(y_test, y_pred)
# # print("Accuracy Score:", accuracy)
# # # measure accuracy
# # plot_confusion_matrix(logisticRegr, x_test, y_test)
# # plt.show()
