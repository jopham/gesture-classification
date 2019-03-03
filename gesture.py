"""
RF - gesture classification model
"""


########################
# IMPORT
########################
import numpy as np
import pandas as pd
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import time

########################
# DATA PREPARATION
########################

# Read in the original dataset
all_data = pd.read_csv('original_copy.csv', header = None)

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