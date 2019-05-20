"""
Purpose: Try a few ML models to classify hand gestures
"""

########################
# IMPORT
########################
import numpy as np
import pandas as pd
from scipy.stats import randint
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import catboost
from catboost import CatBoostClassifier


########################
# DATA PREPARATION
########################

# Read in the original dataset
all_data = pd.read_csv('og_gesture.csv', header=None)

# Choose a seed
seed = 12345

# Create variable names (lables) and target name
count = 1
labellist = []

while count <= 128:
    label_str = "electrode_"
    label_str += str(count)
    labellist.append(label_str)
    count += 1

labellist.append("gesture")
del(count, label_str)

# Change column names
all_data.columns = labellist

# Identify inputs and the target variable
X = all_data[labellist[:-1]]
y = all_data["gesture"]

# Check headers
#print(x.columns.values)

# Split into training and validation (test) datasets. Ratio: 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

########################
# 1. RANDOM FOREST
########################

#Basic model
clf_basic = RandomForestClassifier(n_estimators=100, random_state=seed)
clf_basic.fit(X_train, y_train)
y_pred = clf_basic.predict(X_test)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of the basic model: ", accuracy)

# Parameter tuning - Randomized search
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"n_estimators": [200, 300, 500],
              "max_depth": np.arange(10, 75),
              "max_features": randint(1, 12),
              "min_samples_leaf": [0.001],
              "min_samples_split": randint(4, 10),
              "n_jobs": [-1],
              "criterion": ["gini", "entropy"]}

# Instantiate a RF classifier: clf
clf = RandomForestClassifier(random_state=seed)

# Instantiate the RandomizedSearchCV object: clf_cv
my_cv = 5
clf_cv = RandomizedSearchCV(clf, param_dist, cv=my_cv)

# Fit it to the data
clf_cv.fit(X_train, y_train)

# Predict values on the test set
y_pred = clf_cv.predict(X_test)

# Print the tuned parameters and score
print("Tuned Random Forest Parameters: {}".format(clf_cv.best_params_))
print("Best score is {}".format(clf_cv.best_score_))


########################
# 2. GRADIENT BOOSTING
########################

# Define a basic model using catboost
model = CatBoostClassifier(iterations=50,
                           random_seed=seed,
                           learning_rate=0.1,
                           loss_function='MultiClass')

model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          verbose=20)

print('Model is fitted: ' + str(model.is_fitted()))
print('Model params:')
print(model.get_params())

# Tune your model
model = CatBoostClassifier(iterations=150,
                           random_seed=seed,
                           learning_rate=0.1,
                           loss_function='MultiClass')

model.fit(X_train, y_train,
          eval_set=(X_test, y_test),
          verbose=20)
