# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:04:11 2023

@author: Frede
"""

import unit_commitment as uc
import sample_gen as sg
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import plot

#%% Step 3.1: Create samples
cwd = os.getcwd()
demand = pd.read_csv(cwd + '/system_data/demand.csv')
demand = pd.DataFrame(demand)
demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])

""" Task: More samples and shuffle the way each day is generated """
""" Task: Create samples which activates line constraints """
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

X_model = np.array(X).reshape(24*n_samples,91)
y_G_model = np.array(y_G).reshape(54,24*n_samples)
y_L_model = np.array(y_L).reshape(186,24*n_samples)

#%% Save sample training and test data for future quick runs/reruns
X_file = cwd + "/generated_samples/X_model.csv"
y_G_file = cwd + "/generated_samples/y_G_model.csv"
y_L_file = cwd + "/generated_samples/y_L_model.csv"

pd.DataFrame(X_model).to_csv(X_file)
pd.DataFrame(y_G_model).to_csv(y_G_file)
pd.DataFrame(y_L_model).to_csv(y_L_file)

#%% Load sample training and test data
X_model = np.asarray(pd.read_csv(X_file,index_col=0))
y_G_model = np.asarray(pd.read_csv(y_G_file,index_col=0))
y_L_model = np.asarray(pd.read_csv(y_L_file,index_col=0))

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
    classifiers = [{} for clf in clfs]
    predictions = [{} for clf in clfs]
    
    for g in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(X, y[g], test_size=0.2, shuffle=True, random_state=42)    
        
        for i in range(len(clfs)):
            clf = clfs[i]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test,y_pred)
            accuracies[i][str(target + ":" + str(g))] = accuracy
            classifiers[i][str(target + ":" + str(g))] = clf
            predictions[i][str(target + ":" + str(g))] = y_pred
    
    return accuracies, classifiers, predictions, X_train, X_test

### Predict generator status ###
clfs = [svm.SVC(), RandomForestClassifier()]
acc_G, clf_G, y_pred_G, X_train, X_test = classifiers(X_3days, y_G_model[:,1:-1], clfs, target="G")
acc_G_svm, acc_G_rf = acc_G[0], acc_G[1]

### Step 4 - Predict active constraints ###
clfs = [svm.SVC(), RandomForestClassifier()]
acc_L, clf_L, y_pred_L, X_train, X_test = classifiers(X_3days, y_L_model[:,1:-1], clfs, target="L")
acc_L_svm, acc_L_rf = acc_L[0], acc_L[1]

""" Task: Other classifiers? SVM always predicts 0 currently,
    so something definitely needs to change - or is that the conclusion for
    our linear solver? """

#%% Step 5 - evaluation
# Evaluation of model accuracies:
plot.accComparison(acc_G_svm, acc_G_rf)
plot.accComparison(acc_L_svm, acc_L_rf)


# Evaluation of unit commitment feasibility:
""" Task: Evaluation should also be in terms of actually performing the
    unit commitment. So use the predicted values of 24 subsequent hours
    and see if it is feasible maybe. """

# Maybe plot potential relation between cost of gen and accuracy of prediction

#%% Step 6
""" Task: Evaluation of runtime of unit commitment problem when initialing
    with predicted 'b'-variables. """
    
""" Task: Evaluation of runtime of unit commitment problem when trimming
    problem by removing predicted inactive line constraints. """


#%% Step 7
y_pred_G_svm, y_pred_G_rf = y_pred_G[0], y_pred_G[1]

plot.pca_plot(X_train, X_test, y_pred_G_rf, gen='G:37', pc = [1,2])

plot.pca_plot(X_train, X_test, y_pred_G_svm, gen='G:37', pc = [1,2])







