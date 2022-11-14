from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit

le = preprocessing.LabelEncoder()
dataset = pd.read_csv("NYSE.csv")
X = dataset.drop(['Symbol', 'Start', 'End', 'Label'], axis=1)
y = dataset['Label']

dataset2 = pd.read_csv("NYSE.csv")
X2 = dataset2.drop(['Symbol', 'Start', 'End', 'Label'], axis=1)
y2 = dataset2['Label']
# split the data into training and test set
# X_train = X
# X_test = X2
# y_train = y
# y_test = y2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
ohe = OneHotEncoder(sparse=False)
stock = ohe.fit_transform(X_train)

# get the categorical and numeric column names
num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
print(num_cols)
# pipeline for numerical columns
num_pipe = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler()
)
# pipeline for categorical columns
cat_pipe = make_pipeline(
    SimpleImputer(strategy='constant', fill_value='N/A'),
    OneHotEncoder(handle_unknown='ignore', sparse=False)
)

# combine both the pipelines
full_pipe = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# build the model
logreg = make_pipeline(
    full_pipe, LogisticRegression(max_iter=1000, random_state=42))

# train the model
logreg.fit(X_train, y_train)

# make predictions on the test set
y_pred = logreg.predict(X_test)
# print(y_pred)
# measure accuracy
score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", score)
