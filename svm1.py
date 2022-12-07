# svm.py
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
# import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score 
from sklearn.utils import shuffle
# >> FEATURE SELECTION << #
# def remove_correlated_features(X):
# def remove_less_significant_features(X, Y):
# # >> MODEL TRAINING << #
def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)
    
    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    print("x: ", X_batch)
    print("y: ", Y_batch)
    print("w: ", W)
    print("dw: ", dw)
    print("dot: ", np.dot(X_batch, W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)  # average
    return dw

def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    # stochastic gradient descent
    for epoch in range(1, max_epochs): 
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
            
    return weights

def init():
    data = pd.read_csv('training_set.csv', index_col=[0])
    data = data.copy()

    # Convert label UP=1 and DOWN=0
    data['Label'] = data['Label'].apply(lambda x: 1 if x == "UP" else 0)

    data['Sector'].replace(['Consumer Discretionary', 'Health Care', 'Finance', 'Technology',
                            'Industrials', 'Miscellaneous', 'Utilities', 'Telecommunications',
                            'Real Estate', 'Energy', 'Consumer Staples', 'Basic Materials'],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

    X = data.drop(['Symbol', 'Start', 'End', 'Label'], axis=1).copy()
    y = data['Label'].copy()
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

    # train the model
    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # testing the model on test set
    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i])) #model
        y_test_predicted = np.append(y_test_predicted, yp)
        print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
        print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
        print("precision on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))

# set hyper-parameters and call init
# hyper-parameters are normally tuned using cross-validation
# but following work good enough
reg_strength = 10000 # regularization strength
learning_rate = 0.000001
init()
