import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
from numpy import linalg
from sklearn import svm
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
# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False


def linear_kernel(x1, x2, x3=None):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * np.linalg.norm(x - y) ** 2)


class SVM(object):
    def __init__(self, kernel=gaussian_kernel, param=None, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)
        self.param = param
        if self.param is not None:
            self.param = float(self.param)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j], self.param)

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))
        A = A * 1.0
        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def predict_set(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv, self.param)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.predict_set(X))


if __name__ == "__main__":
    import pylab as pl

    data = pd.read_csv("training_set_0.csv")
    data.reset_index()
    data = data.copy()
    # Convert label UP=1 and DOWN=0
    data['Label'] = data['Label'].apply(lambda x: 1 if x == "UP" else -1)

    data['Sector'].replace(['Consumer Discretionary', 'Health Care', 'Finance', 'Technology',
                            'Industrials', 'Miscellaneous', 'Utilities', 'Telecommunications',
                            'Real Estate', 'Energy', 'Consumer Staples', 'Basic Materials'],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)
    test_data = pd.read_csv("test_set_0.csv")
    test_data.reset_index()
    test_data = test_data.copy()
    # Convert label UP=1 and DOWN=-1
    test_data['Label'] = test_data['Label'].apply(
        lambda x: 1 if x == "UP" else -1)

    test_data['Sector'].replace(['Consumer Discretionary', 'Health Care', 'Finance', 'Technology',
                                'Industrials', 'Miscellaneous', 'Utilities', 'Telecommunications',
                                 'Real Estate', 'Energy', 'Consumer Staples', 'Basic Materials'],
                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

    X = data.drop(['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
    y = data['Label'].copy()
    x_test_data = test_data.drop(
        ['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
    y_test_data = test_data['Label'].copy()
    y_test_data = np.hstack(y_test_data)

    # Standardizing the feature values
    sc_x = StandardScaler()
    x_test_data = sc_x.fit_transform(x_test_data)
    x_test_data = np.vstack(x_test_data)

    ####################################################################
    # Use this section for test set with the kernel and the hyperparameter
    ##################################################################
    # x_train, x_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42)
    # # Standardizing the feature values
    # sc_x = StandardScaler()
    # x_train = sc_x.fit_transform(x_train)
    # x_train = np.vstack(x_train)

    # x_test = sc_x.transform(x_test)
    # x_test = np.vstack(x_test)

    # y_train = np.hstack(y_train)
    # y_test = np.hstack(y_test)

    ############ Change the kernel and C value ################
    # clf = SVM(gaussian_kernel, param=10, C=10)
    # clf.fit(x_train, y_train)

    ################ Validation set #####################
    # y_pred = clf.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Training Accuracy Score:", accuracy)
    ############ Confusion Matrix #################
    #     ax = plt.axes()
    #     cf_matrix = metrics.confusion_matrix(y_test, y_pred)
    #     s = sn.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
    #     s.set(xlabel='Predicted Label', ylabel='True Label')
    #     ax.set_title("Accuracy = " + str(accuracy))
    #     plt.show()
    ######################## Test set ###########################
    # y_pred = clf.predict(x_test_data)
    # accuracy = accuracy_score(y_test_data, y_pred)
    # print("Test Accuracy Score:", accuracy)
    ############ Confusion Matrix #################
    #     ax = plt.axes()
    #     cf_matrix = metrics.confusion_matrix(y_test_data, y_pred)
    #     s = sn.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
    #     s.set(xlabel='Predicted Label', ylabel='True Label')
    #     ax.set_title("Accuracy = " + str(accuracy))
    #     plt.show()
    ##################################################################

    # List the hyperparameter for your kernel
    hyperparameter = [0.01, 0.1, 1, 3]
    c_value = [0.01, 0.1, 1, 10]  # List the c for the SVM
    kf = KFold(n_splits=5)
    for i in hyperparameter:
        for c in c_value:
            acc = []
            f1 = []
            precision = []
            recall = []
            print("Using parameter {} and C={}".format(i, c))
            for train_index, test_index in kf.split(X):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                sc_x = StandardScaler()
                x_train = sc_x.fit_transform(x_train)
                x_train = np.vstack(x_train)
                x_test = sc_x.transform(x_test)
                x_test = np.vstack(x_test)
                y_train = np.hstack(y_train)
                y_test = np.hstack(y_test)
                ########################################
                # Change the kernel here
                ########################################
                clf = SVM(rbf_kernel, param=i, C=c)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                acc.append(accuracy)
                f1.append(metrics.f1_score(y_test, y_pred))
                precision.append(metrics.precision_score(y_test, y_pred))
                recall.append(metrics.recall_score(y_test, y_pred))

            print("Accuracy-Score Min: {} Max: {} Avg: {}".format(np.min(acc),
                  np.max(acc), np.average(acc)))
            print("F1-Score Min: {} Max: {} Avg: {}".format(np.min(f1),
                  np.max(f1), np.average(f1)))
            print("Precision-Score Min: {} Max: {} Avg: {}".format(np.min(precision),
                                                                   np.max(precision), np.average(precision)))
            print("Recall-Score Min: {} Max: {} Avg: {}".format(np.min(recall),
                                                                np.max(recall), np.average(recall)))
