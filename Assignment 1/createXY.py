import pandas as pd
import os, csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv


#%% Feature data and actual wind power production: 
cwd = os.getcwd()

""" Load data files from folder """
temp_dir = os.path.join(cwd, 'ClimateData/data.csv')
with open(temp_dir) as data_file:
    data = pd.read_csv(data_file)
    data.set_index('HourUTC', inplace=True)

""" Load actual wind power from cwd """
temp_dir = os.path.join(cwd, 'Actual wind power.csv')
with open(temp_dir) as data_file:
    df = pd.read_csv(data_file,sep=";")
    plt.plot(df['Actual'])
    df["HourUTC"] = df['Date'].astype(str) +" "+ df["Time"]
    df['HourUTC'] = pd.to_datetime(df['HourUTC'])
    df = df[['HourUTC','Actual']]
    df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
    df = df.drop_duplicates('HourUTC',keep='first')
    df.set_index('HourUTC', inplace=True)
    df1Jan = np.mean([df['2022-01-01 01:00']['Actual'].values[0],df['2021-12-31 23:00']['Actual'].values[0]])
    line = pd.DataFrame({"Actual": df1Jan}, index=[datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')])
    df = df.append(line, ignore_index=False)
    df = df.sort_index()
    df = df.iloc[0:len(df['Actual'])-1,:] 

#%% Standardization 
attributeNames = np.asarray(data.columns)
dfs = data.copy()
mu_dfs = np.mean(dfs[attributeNames])
std_dfs = np.std(dfs[attributeNames])

for i in range(0,len(attributeNames)):
    dfs[attributeNames[i]] = (dfs[attributeNames[i]]-mu_dfs[i])/std_dfs[i]


#%% Formatting
y = np.array(df['Actual'].values)
x1 = np.ones(len(y))
cols = dfs.columns
X = np.matrix([x1,dfs[cols[0]],dfs[cols[1]],dfs[cols[2]]]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

#%% First sample data
size = 300
X_slice = X[0:size]
y_slice = y[0:size]

X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=0.4, shuffle=False)

# Closed form linear regression:
beta = np.array(inv(X_train.T @ X_train) @ X_train.T @ y_train).reshape(-1)
y_pred = np.array(X_test @ beta).reshape(-1)
mse = mean_squared_error(y_test, y_pred) # 0.028742528161411984





