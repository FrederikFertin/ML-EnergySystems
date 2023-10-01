import pandas as pd
import os, csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv
from model1 import closed_form_fit,closed_form_predict

#%% Feature data and actual wind power production:
cwd = os.getcwd()

""" Load actual wind power from cwd """
temp_dir = os.path.join(cwd, 'Actual wind power.csv')
df = pd.read_csv(temp_dir,sep=";")

### Create time index to match feature data
df["HourUTC"] = df['Date'].astype(str) +" "+ df["Time"]
df['HourUTC'] = df['HourUTC'].apply(lambda x: str(x.replace('.','-')))
df['HourUTC'] = pd.to_datetime(df['HourUTC'], format='%d-%m-%Y %H:%M')
df = df[['HourUTC','Actual']]
df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
df = df.drop_duplicates('HourUTC',keep='first')
df.set_index('HourUTC', inplace=True)

### Sort out DST issues with duplicate values and missing observation (2022-01-01 00:00)
df1Jan = np.mean([df.loc['2022-01-01 01:00']['Actual'],df.loc['2021-12-31 23:00']['Actual']])
line = pd.DataFrame({"Actual": df1Jan}, index=[datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')])
df = df.append(line, ignore_index=False)
df = df.sort_index()
df = df.iloc[0:len(df['Actual'])-1,:]
""" Power production is already normalized """


""" Load data files from folder """
temp_dir = os.path.join(cwd, 'ClimateData/data.csv')
data = pd.read_csv(temp_dir)
data.set_index('HourUTC', inplace=True)

#%% Step 2: Creating the training and test datasets
power_prod_prev_36 = np.array(np.append(df['Actual'].values[0:36],df['Actual'].values[0:len(df.Actual)-36]),dtype=object)
wind_cubed = np.array((data['wind_speed [m/s]']**3).values)
windXpressure = np.array((data['wind_speed [m/s]']**3).values*data['pressure [hPa]'].values)
data['wind_cubed'] = wind_cubed
data['wind_energy'] = windXpressure
data['ones'] = 1

#%% Standardization
attributeNames = np.asarray(data.columns)
dfs = data.copy()
mu_dfs = np.mean(dfs[attributeNames])
std_dfs = np.std(dfs[attributeNames])

for i in range(0,len(attributeNames)):
    dfs[attributeNames[i]] = (dfs[attributeNames[i]]-mu_dfs[i])/std_dfs[i]


# Formatting
y = np.array(df['Actual'].values)
x1 = np.ones(len(y))
cols = dfs.columns
X = np.matrix([x1,dfs[cols[0]],dfs[cols[1]],dfs[cols[5]]]).T
#X = np.matrix([x1,dfs[cols[0]],dfs[cols[1]],dfs[cols[2]],dfs[cols[3]],dfs[cols[4]]]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

#%% Step 3: First sample data
sizes = [100,1000,len(X)]
feat = [['ones','wind_speed [m/s]'],['ones','wind_speed [m/s]','temperature [C]'],['ones','wind_cubed'],['ones','temperature [C]']]
mse_list = []

for features in feat:
    mse_list.append(features)
    for size in sizes:
        X = np.array(data[features])
        #X = np.matrix([x1,data[cols[0]],data[cols[1]],data[cols[2]]]).T
        X_slice = X[0:size]
        y_slice = y[0:size]

        X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=0.4, shuffle=False)

        # Closed form linear regression:
        beta = closed_form_fit(X_train, y_train)
        y_pred = closed_form_predict(beta, X_test)
        mse = mean_squared_error(y_test, y_pred)  # 0.028742528161411984
        mse_list.append(mse)
print(mse_list)

#%% Step 4: Non-linear regression
wind_cubed = np.array((data['wind_speed [m/s]']**3).values)
windXpressure = np.array((data['wind_speed [m/s]']**3).values*data['pressure [hPa]'].values)
dfs['wind_cubed'] = wind_cubed
dfs['wind_energy'] = windXpressure


