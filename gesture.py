"""
RF - gesture classification model
"""

########################
# IMPORT
########################
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import time

########################
# DATA PREPARATION
########################

# Read in the original dataset
all_data = pd.read_csv('og_gesture.csv', header = None)

# Choose a seed
seed = 12345
np.random.seed(seed)

# Create variable names (lables) and target name
count = 1
labellist = []

while count <= 128:
    string = "electrode_"
    string += str(count)
    labellist.append(string)
    count += 1

labellist.append("gesture")
del(count, string)

# Change column names
all_data.columns = labellist

# Add an ID column
all_data.insert(0, 'id', range(1, 1 + len(all_data)))

# Identify inputs and the target variable
x = all_data[labellist[:-1]]
y = all_data["gesture"]

# Split into training and validation (test) datasets. Ratio: 70/30
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = seed)

########################
# 1. RANDOM FOREST 
########################

#Basic model
clf_basic = RandomForestClassifier(n_estimators = 100, random_state = seed)
clf_basic.fit(x_train, y_train)
y_pred = clf_basic.predict(x_test)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of the basic model: ", accuracy)

# PARAMETER TUNING

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"n_estimators": [500], # 500, 1000], #[100, 200],
              "max_depth": [72], #np.arange(10, 75),
              "max_features": [3], #randint(1, 12), # np.arange(1, 12), #randint(1, 12)
              "min_samples_leaf": [0.001], #randint(0.01, 0.1), #[100], #[100, 200, 300, 400, 500],
              "min_samples_split": [4],
              "n_jobs": [-1],
              "criterion": ["gini"]} #"gini", "entropy"]}

print ('start fitting')             # Change this later to write to log
clf = RandomForestClassifier(n_estimators = 500, min_samples_leaf=0.00075, max_features=3, 
                             criterion = "gini", 
                             random_state = seed, n_jobs = -1) # max depth not needed,

# Fit the RF model
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

y_pred_train = clf.predict(x_train)
accuracy = metrics.accuracy_score(y_train, y_pred_train)
print("Accuracy train: ", accuracy)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy on the test set: ", accuracy)       

# Save optimal parameters and accuracy

########################
# 2. XGBOOST
########################

# Convert to Dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y
                           
# "Basic" model
xg_reg = xgb.XGBClassifier(max_features='sqrt', subsample=0.8, random_state=seed)
                           
# Parameter tuning
# Grid Search (you can try Random Search too)
xgb_params = [{'n_estimators': [10, 100]},
              {'learning_rate': [0.1, 0.01, 0.5]}]
xgb_gsearch = GridSearchCV(estimator = gbm, param_grid = parameters, scoring='accuracy', cv = 3, n_jobs=-1)
#xgb_gsearch = grid_search.fit(x_train, y_train)