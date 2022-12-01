import numpy as np
from sklearn.metrics import accuracy_score

"""
Optimization of Logistic Regression 
We want to find the best values for our weights
1. Gradient descent 
2. Maximum likelihood estimation
"""


class LogisticRegression():
    def __init__(self, fit_intercept=True, verbose=True):
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    # Cost function finds the error between actual value and predicted value
    # Hypothesis is nonconvex function, meaning there are chances of
    # finding the local minima and avoiding the global minima
    # Smoothened the curve with the help of log and our cost function will look like this
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    """
    Sigmoid function predict or classify a given input

    z = (theta^t)(x), where theta is the weight
    sigmoid(z) = 1 / (1 + e^-(z))

    Gradient descent - derivative of the loss function with respect to the weights
    The goal is to minimize the loss (cost function) by increasing or decreasing the weights (fitting)
    The weights are updated by subtracting the gradient times the learning rate
    """

    def fit_sigmoid(self, X, y, learning_rate=0.01, num_iter=1000):
        if self.fit_intercept:
            X = self.add_intercept(X)

        # weights initialization
        self.weights = np.zeros(X.shape[1])
        self.loss_history = []
        self.iter_history = []

        for i in range(num_iter):
            z = np.dot(X, self.weights)
            h = 1 / (1 + np.exp(-z))
            gradient = np.dot(X.T, (h - y)) / y.shape[0]
            self.weights -= learning_rate * gradient

            if(self.verbose == True and i % 500 == 0):
                z = np.dot(X, self.weights)
                h = 1 / (1 + np.exp(-z))
                loss = self.loss(h, y)
                self.loss_history.append(loss)
                self.iter_history.append(i)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return 1 / (1 + np.exp(-np.dot(X, self.weights)))

    def predict(self, X):
        return self.predict_prob(X).round()
