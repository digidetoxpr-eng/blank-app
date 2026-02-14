#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Programming Assignment
# ## Comparing Logistic regression Models 
# 
# **Student Name:** ____prathap ramados_______________  
# **Student ID:** ___2025AA05488________________  
# **Date:** _____11th Jan 2026____________
# 
# ---
# 
# ---

# In[2]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#decision tree classifier

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# K Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#Naive bayes classigied - Gaussian or multinomial
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, classification_report



#Ensemble Model - XGBoost
#!pip3 install xgboost
from xgboost import XGBRFClassifier

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
print('✓ Libraries imported successfully')
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[3]:


# save python models using job lib 

import joblib


def saveModel(filename,model):
    joblib.dump(model, filename)

def loadModel(filename):
    loaded_model = joblib.load(filename)
    return loaded_model



# ## Section 1: Dataset Selection and Loading
# 
# **Requirements:**
# - ≥500 samples
# - ≥5 features
# - Public dataset (UCI/Kaggle)
# - Regression OR Classification problem

# In[5]:


def loadData(filename):
    # TODO: Load your dataset
    data  =pd.read_csv('Airline_customer_satisfaction.csv')
    # Dataset information (TODO: Fill these)
    dataset_name = "Airline Customer Satisfaction"  
    dataset_source = "Kaggle"  # e.g., "UCI ML Repository"
    n_samples = data.shape[0]      # Total number of rows
    n_features = data.shape[1]    # Number of features (excluding target)
    problem_type = "Logistic Regression"  # "regression" or "binary_classification" or "multiclass_classification"

    # Problem statement (TODO: Write 2-3 sentences)
    problem_statement = "Predicting customer satisfaction from an airline industry survey measuring customer satisfaction"

    # Primary evaluation metric (TODO: Fill this)
    primary_metric =   "accuracy"

    # Metric justification (TODO: Write 2-3 sentences)
    metric_justification = """
    predicting Airline customer satisfaction level as a binary classification. binary classification with balanced set of classes in available data 
    means Accuracy can be a good measure of metric.
    "
    """
    print(f"Dataset: {dataset_name}")
    print(f"Source: {dataset_source}")
    print(f"Samples: {n_samples}, Features: {n_features}")
    print(f"Problem Type: {problem_type}")
    print(f"Primary Metric: {primary_metric}")

    print(data.shape)

    data.dtypes
    print(data.describe())
    return data


# ## Section 2: Data Preprocessing
# 
# Preprocess your data:
# 1. Handle missing values
# 2. Encode categorical variables
# 3. Split into train/test sets
# 4. Scale features

# In[7]:


def preProcess(data):

    # fix any data issues 
    data = data.drop('Arrival Delay in Minutes',axis=1)
    data = data.drop_duplicates()

    #get count of null values - if any 
    data.isnull().sum()


    #replace binary string values with boolean values 
    #data  = data[:5000]
    data['satisfaction'] =data['satisfaction'].replace('satisfied',1)
    data['satisfaction'] =data['satisfaction'].replace('dissatisfied',0)
    data['Customer Type'] =data['Customer Type'].replace('Loyal Customer',1)
    data['Customer Type'] =data['Customer Type'].replace('disloyal Customer',0)

    data['Type of Travel'] =data['Type of Travel'].replace('Personal Travel',0)
    data['Type of Travel'] =data['Type of Travel'].replace('Business travel',1)
    data['Type of Travel'] =data['Type of Travel'].astype(int)


    data['Class'] =data['Class'].replace('Eco',0)
    data['Class'] =data['Class'].replace('Eco Plus',1)
    data['Class'] =data['Class'].replace('Business',2)

    #find outliers

    #data.boxplot(column=['price'])

    #plt.hist(data['price'],bins=20)
    #plt.xlabel("value")
    #plt.ylabel("frequency")
    #plt.show()
    #data.head()


    data.dtypes

    print(data.describe())
    return data


# In[8]:


def prepareTestData(data):
    #get input columns and target column ready 
    y=data['satisfaction']
    x=data.drop('satisfaction',axis=1)
    return x,y


# In[9]:


def splitTrainTest(x,y):


    # TODO: Train-test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = 0.2, random_state=42)
    print("Train data shape of X = % s and Y = % s : "%(x_train.shape, y_train.shape))

    print("Test data shape of X = % s and Y = % s : "%(x_test.shape, y_test.shape))

    # TODO: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)

    #print(X_train_scaled)
    X_test_scaled = scaler.transform(x_test)

    # Fill these after preprocessing
    train_samples = x_train.shape[0]       # Number of training samples
    test_samples = x_test.shape[0]        # Number of test samples
    train_test_ratio = 0.2  # e.g., 0.8 for 80-20 split

    print(f"Train samples: {train_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Split ratio: {train_test_ratio:.1%}")


    #get training /test data split and ready 
    return x_train, x_test, y_train, y_test


# In[10]:


def getLogisticRegressionModel(x_train,y_train):
    # Create an instance of the LogisticRegression model
    # 'liblinear' solver is good for small datasets, 'lbfgs' is default for larger ones
    model = LogisticRegression(solver='lbfgs', random_state=0)

    # Fit (train) the model with the training data
    model.fit(x_train, y_train)
    return model



# In[ ]:






# In[11]:


def getDecisionTreeClassifier(x_train,y_train):
    # Create a Decision Tree Classifier instance with a maximum depth of 3
    model = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=0)

    # Train the model using the training data
    model.fit(x_train, y_train)
    return model



# In[12]:


def getKNearestNeighbour(x_train,y_train):
    k_value = 5 # Choose an appropriate value for k
    model = KNeighborsClassifier(n_neighbors=k_value)

    # Fit the model to the scaled training data
    model.fit(x_train, y_train)
    return model



# In[13]:


def getGaussianNB(x_train,y_train):
    model = GaussianNB()

    # 4. Train (fit) the model
    model.fit(x_train, y_train)
    return model



# In[14]:


#Ensemble model - randm forest

def getRandomForest(x_train,y_train):
    # n_estimators is the number of trees in the forest (default is 100)
    model = RandomForestClassifier(n_estimators=100, random_state=0)

    #Train the model
    model.fit(x_train, y_train)
    return model


# In[15]:


def getXGBoost(x_train,y_train):
    # Key parameters for Random Forest behavior in XGBoost:
    # n_estimators: number of trees in the forest (e.g., 100)
    # subsample: fraction of samples used for each tree (e.g., 0.9 for bagging)
    # colsample_bynode: fraction of features used for each tree/node (e.g., 0.2 for random subspace)
    model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2, random_state=42)

    # 3. Fit the model
    model.fit(x_train,y_train)
    return model


# In[16]:


def printRegressionMetrics(y_test,y_pred,modelName):

    metricsDict={}
    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")
    precision=precision_score(y_test,y_pred)
    print(f"Precision score:{precision}")
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"Accuracy Score: {accuracy}")
    # To get the count of correct predictions
    accuracy_count = accuracy_score(y_test, y_pred, normalize=False)
    print(f"Number of correct predictions: {accuracy_count}")

    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    metricsDict.update({"name": modelName, "accuracy": accuracy,"precision":precision,"auc_score": auc_score,"recall": recall,"f1_score": f1,"mcc_score": mcc})

    return metricsDict


# In[17]:


def printConfusionMatrix(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    return cm


# In[18]:


#execution logic 


#load Data 
data = loadData("test.csv")

#PreProcess Data
data = preProcess(data)
X,Y= prepareTestData(data)

#train /test split
x_train, x_test, y_train, y_test = splitTrainTest(X,Y)

#get binary classification model instantiated and run prediction
model = getLogisticRegressionModel(x_train,y_train)
y_pred = model.predict(x_test)
print(printRegressionMetrics(y_test,y_pred,"LogisticRegression"))
saveModel("LogisticRegression.sav",model)
model = getDecisionTreeClassifier(x_train,y_train)
y_pred = model.predict(x_test)
print(printRegressionMetrics(y_test,y_pred,"DecisionTree"))          
saveModel("DecisionTree.sav",model)
model = getKNearestNeighbour(x_train,y_train)
y_pred = model.predict(x_test)
print(printRegressionMetrics(y_test,y_pred,"KNN"))          
saveModel("KNN.sav",model)
model = getGaussianNB(x_train,y_train)
y_pred = model.predict(x_test)
print(printRegressionMetrics(y_test,y_pred,"Gaussian"))          
saveModel("Gaussian.sav",model)
model = getRandomForest(x_train,y_train)
y_pred = model.predict(x_test)
print(printRegressionMetrics(y_test,y_pred,"RandomForest"))          
saveModel("RandomForest.sav",model)


model = getXGBoost(x_train,y_train)
y_pred=model.predict(x_test)
print(printRegressionMetrics(y_test,y_pred,"XGBoost"))
saveModel("xgboost.sav",model)

print(printConfusionMatrix(y_test,y_pred))



# In[ ]:




