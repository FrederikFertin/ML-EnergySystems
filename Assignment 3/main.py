# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:04:11 2023

@author: Frede
"""

import unit_commitment as uc
import sample_gen as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import plot

#%% Step 3.1: Create samples
demand = pd.read_csv('system_data/demand.csv')
demand = pd.DataFrame(demand)
demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])

""" Task: More samples and shuffle the way each day is generated """
n_samples = 20

multis = np.linspace(0.8, 1.2, n_samples)

demands = [demand*multi for multi in multis]

# = sg.generate_samples(multipliers=multis)

#%% Step 3: Create data sets 
X = []
y_G = []
y_L = []

for i, sample in enumerate(demands):
    print("Sample ", i)
    X_sample, y_G_sample, y_L_sample = uc.uc(sample)
    X.append(X_sample)
    y_G.append(y_G_sample)
    y_L.append(y_L_sample)

""" Task: Save X, y, and cong_lines in json files as dictionaries so we do not need to rerun """

X_model = np.array(X).reshape(24*n_samples,91)
y_G_model = np.array(y_G).reshape(54,24*n_samples)
y_L_model = np.array(y_L).reshape(186,24*n_samples)

#%% Create altered data set 
# Initializing lists 
X_3days = list()

# Create sequences of 3 hours for X
for h in range(1,len(X_model)-1):
    X_3days.append(X_model[h-1:h+2].flatten())

# Change the type 
X_3days = np.asarray(X_3days)

#%% Step 4: Classification
def classifiers(X, y, clfs=[svm.SVC()], target="G"):
    accuracies = [{} for clf in clfs]
    for g in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(X, y[g], test_size=0.2, shuffle=True, random_state=42)    
        
        for i in range(len(clfs)):
            clf = clfs[i]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            rmse = accuracy_score(y_test,y_pred)
            accuracies[i][str(target + ":" + str(g))] = rmse
        
    return accuracies

#%% Predict generators 
clfs = [svm.SVC(), RandomForestClassifier()]
acc_G = classifiers(X_3days, y_G_model[:,1:-1], clfs, target="G")
acc_G_svm, acc_G_rf = acc_G[0], acc_G[1]

#%% Predict active constraints 
clfs = [svm.SVC(), RandomForestClassifier()]
acc_L = classifiers(X_3days, y_L_model[:,1:-1], clfs, target="L")
acc_L_svm, acc_L_rf = acc_L[0], acc_L[1]

""" Task: Predict active constraints """

""" Task: Other classifiers? """

#%%
plot.accComparison(acc_G_svm, acc_G_rf)

plot.accComparison(acc_L_svm, acc_L_rf)


# Maybe plot potential relation between cost of gen and accuracy of prediction












