import numpy as np
from sklearn.metrics import accuracy_score

"""
Optimization of Logistic Regression 
We want to find the best values for our weights
1. Gradient descent 
2. Maximum likelihood estimation
"""


class LogisticRegression():
    def __init__(self, fit_intercept=True, verbose=False):
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    """
    Sigmoid function predict or classify a given input

    z = (theta^t)(x), where theta is the weight
    sigmoid(z) = 1 / (1 + e^-(z))
    """

    def fit_sigmoid(self, X, y, learning_rate=0.01, num_iter=10000):
        if self.fit_intercept:
            X = self.add_intercept(X)

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        for i in range(num_iter):
            z = np.dot(X, self.weights)
            h = 1 / (1 + np.exp(-z))
            gradient = np.dot(X.T, (h - y)) / y.shape[0]
            self.weights -= learning_rate * gradient

            z = np.dot(X, self.weights)
            h = 1 / (1 + np.exp(-z))
            loss = self.loss(h, y)

            if(self.verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    """
    Gradient descent - derivative of the loss function with respect to the weights
    The goal is to minimize the loss by increasing or decreasing the weights (fitting)
    The weights are updated by subtracting the gradient times the learning rate
    """

    def fit_mle(self, X, y, learning_rate=0.01, num_iter=10000):
        if self.fit_intercept:
            X = self.add_intercept(X)

        # weights initialization
        self.weights = np.zeros(X.shape[1])

        for i in range(num_iter):

            z = np.dot(X, self.weights)
            h = 1 / (1 + np.exp(-z))
            ll = np.sum(y*z - np.log(1 + np.exp(z)))
            gradient = np.dot(X.T, (y-h))
            self.weights += learning_rate * gradient

            z = np.dot(X, self.weights)
            h = 1 / (1 + np.exp(-z))
            ll = np.sum(y*z - np.log(1 + np.exp(z)))
            loss = self.loss(ll, y)

            if(self.verbose == True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return 1 / (1 + np.exp(-np.dot(X, self.weights)))

    def predict(self, X):
        return self.predict_prob(X).round()
