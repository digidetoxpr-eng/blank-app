"""
Machine Learning - Programming Assignment
Comparing Logistic regression Models

Student Name: ____prathap ramados_______________
Student ID: ___2025AA05488________________
Date: _____11th Jan 2026____________

This file was exported from the notebook 2025aa05488-LRModels.ipynb
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns

# K Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
# Naive bayes classifier - Gaussian or multinomial
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

# Ensemble Model - XGBoost (optional)
# from xgboost import XGBRFClassifier

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
print('✓ Libraries imported successfully')

# save python models using joblib
import joblib


def saveModel(filename, model):
    joblib.dump(model, filename)


def loadModel(filename):
    loaded_model = joblib.load(filename)
    return loaded_model


# Section 1: Dataset Selection and Loading

def loadData(fileName):
    # TODO: Load your dataset
    data = pd.read_csv('Airline_customer_satisfaction.csv')
    # Dataset information (TODO: Fill these)
    dataset_name = "Airline Customer Satisfaction"
    dataset_source = "Kaggle"
    n_samples = data.shape[0]
    n_features = data.shape[1]
    problem_type = "Logistic Regression"

    # Problem statement (TODO: Write 2-3 sentences)
    problem_statement = "Predicting customer satisfaction"

    # Primary evaluation metric (TODO: Fill this)
    primary_metric = ("recall", "accuracy", "rmse", "r2")

    # Metric justification (TODO: Write 2-3 sentences)
    metric_justification = """
    predicting Airline customer satisfaction level as a binary classification
    """
    print(f"Dataset: {dataset_name}")
    print(f"Source: {dataset_source}")
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"Problem Type: {problem_type}")
    print(f"Primary Metric: {primary_metric}")

    print(data.shape)
    print(data.describe())
    return data


# Section 2: Data Preprocessing

def preProcess(data):

    # fix any data issues
    if 'Arrival Delay in Minutes' in data.columns:
        data = data.drop('Arrival Delay in Minutes', axis=1)
    data = data.drop_duplicates()

    # get count of null values - if any
    # data.isnull().sum()

    # Limit to first 5000 rows if dataset is large
    data = data[:5000]

    # Encode target and categorical features
    data['satisfaction'] = data['satisfaction'].replace({'satisfied': 1, 'dissatisfied': 0})
    data['Customer Type'] = data['Customer Type'].replace({'Loyal Customer': 1, 'disloyal Customer': 0})
    data['Type of Travel'] = data['Type of Travel'].replace({'Personal Travel': 0, 'Business travel': 1})
    try:
        data['Type of Travel'] = data['Type of Travel'].astype(int)
    except Exception:
        pass

    data['Class'] = data['Class'].replace({'Eco': 0, 'Eco Plus': 1, 'Business': 2})

    print(data.describe())
    return data


def prepareTestData(data):
    # get input columns and target column ready
    y = data['satisfaction']
    x = data.drop('satisfaction', axis=1)
    return x, y


def splitTrainTest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Train data shape of X = % s and Y = % s : " % (x_train.shape, y_train.shape))
    print("Test data shape of X = % s and Y = % s : " % (x_test.shape, y_test.shape))

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    train_samples = x_train.shape[0]
    test_samples = x_test.shape[0]
    train_test_ratio = 0.2

    print(f"Train samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Split ratio: {train_test_ratio:.1%}")

    return x_train, x_test, y_train, y_test


def getLogisticRegressionModel(x_train, y_train):
    model = LogisticRegression(solver='lbfgs', random_state=0, max_iter=1000)
    model.fit(x_train, y_train)
    return model


def getDecisionTreeClassifier(x_train, y_train):
    model = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=0)
    model.fit(x_train, y_train)
    return model


def getKNearestNeighbour(x_train, y_train):
    k_value = 5
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(x_train, y_train)
    return model


def getGaussianNB(x_train, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    return model


def getRandomForest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(x_train, y_train)
    return model

# XGBoost placeholder (commented out)
# def getXGBoost(x_train, y_train):
#     model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2, random_state=42)
#     model.fit(x_train, y_train)
#     return model


def printRegressionMetrics(y_test, y_pred, modelName):
    metricsDict = {}
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")
    auc_score = roc_auc_score(y_test, y_pred)
    # To get the count of correct predictions
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    print(f"Number of correct predictions: {accuracy_count}")

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    metricsDict.update({"name": modelName, "accuracy": accuracy, "auc_score": auc_score, "recall": recall, "f1_score": f1, "mcc_score": mcc})
    return metricsDict


if __name__ == '__main__':
    # execution logic
    data = loadData("Airline_customer_satisfaction.csv")
    data = preProcess(data)
    X, Y = prepareTestData(data)
    x_train, x_test, y_train, y_test = splitTrainTest(X, Y)

    model = getLogisticRegressionModel(x_train, y_train)
    y_pred = model.predict(x_test)
    print(printRegressionMetrics(y_test, y_pred, "LogisticRegression"))
    saveModel("LogisticRegression.sav", model)

    model = getDecisionTreeClassifier(x_train, y_train)
    y_pred = model.predict(x_test)
    print(printRegressionMetrics(y_test, y_pred, "DecisionTree"))
    saveModel("DecisionTree.sav", model)

    model = getKNearestNeighbour(x_train, y_train)
    y_pred = model.predict(x_test)
    print(printRegressionMetrics(y_test, y_pred, "KNN"))
    saveModel("KNN.sav", model)

    model = getGaussianNB(x_train, y_train)
    y_pred = model.predict(x_test)
    print(printRegressionMetrics(y_test, y_pred, "Gaussian"))
    saveModel("Gaussian.sav", model)

    model = getRandomForest(x_train, y_train)
    y_pred = model.predict(x_test)
    print(printRegressionMetrics(y_test, y_pred, "RandomForest"))
    saveModel("RandomForest.sav", model)

    print('Done')
