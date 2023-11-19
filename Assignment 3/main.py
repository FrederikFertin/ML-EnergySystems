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

demand = pd.read_csv('system_data/demand.csv')
demand = pd.DataFrame(demand)
demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])

""" Task: More samples and shuffle the way each day is generated """
n_samples = 20

multis = np.linspace(0.8, 1.2, n_samples)

demands = [demand*multi for multi in multis]

# = sg.generate_samples(multipliers=multis)

X = []
y = []
cl = []

for i, sample in enumerate(demands):
    print("Sample ", i)
    X_sample, y_sample, cong_lines = uc.uc(sample)
    X.append(X_sample)
    y.append(y_sample)
    cl.append(cong_lines)

""" Task: Save X, y, and cong_lines in json files as dictionaries so we do not need to rerun """

X_model = np.array(X).reshape(24*n_samples,91)
y_model = np.array(y).reshape(54,24*n_samples)

""" Task: Predict active constraints """

svm_accuracies = {}
rf_accuracies = {}

for g in range(len(y_model)):
    X_train, X_test, y_train, y_test = train_test_split(X_model, y_model[g], test_size=0.2, shuffle=True, random_state=42)    
    
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    rmse = accuracy_score(y_test,y_pred)
    svm_accuracies[str("Gen: " + str(g))] = rmse
    
    rf = RandomForestClassifier()
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    rmse = accuracy_score(y_test,y_pred)
    rf_accuracies[str("Gen: " + str(g))] = rmse

""" Task: Other classifiers? """

mean_svm = np.mean(list(svm_accuracies.values()))
mean_rf = np.mean(list(rf_accuracies.values()))
colors = ['blue','green']

plt.plot(rf_accuracies.values(), label="RF",color=colors[0])
plt.axhline(mean_rf,label="Mean RF",color=colors[0], alpha=0.5, linestyle = "--")
plt.plot(svm_accuracies.values(), label="SVM",color=colors[1])
plt.axhline(mean_svm,label="Mean SVM",color=colors[1], alpha=0.5, linestyle = "--")
plt.legend()
plt.title("Binary classifier accuracies for each generator")
plt.show()
plt.plot(y_model.sum(axis=1),label = "Active generator hours")
plt.legend()
plt.show()

# Maybe plot potential relation between cost of gen and accuracy of prediction












