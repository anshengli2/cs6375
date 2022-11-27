
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn import metrics
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logreg import LogisticRegression as CustomLogisticRegression


def confusion_matrix(ytst, y_pred):
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
                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)

X = data.drop(['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
y = data['Label'].copy()

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

warnings.filterwarnings('ignore')
lr = CustomLogisticRegression()
lr.fit_sigmoid(x_train, y_train, learning_rate=0.1, num_iter=10000)
# lr.fit_mle(x_train, y_train)
y_pred = lr.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
confusion_matrix(y_test, y_pred)

# print("\nSklearn's logistic regression")
# logisticRegr = LogisticRegression()
# logisticRegr.fit(x_train, y_train)  # apply scaling on training data
# # make predictions on the test set
# y_pred = logisticRegr.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy Score:", accuracy)
# # measure accuracy
# confusion_matrix(y_pred, y_pred)
