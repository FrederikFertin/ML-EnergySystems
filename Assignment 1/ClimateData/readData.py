# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:23:17 2023

@author: Frede

Request Structure:
    https://dmigw.govcloud.dk/v2/climateData/collections/
    municipalityValue/
    items?
    municipalityId=0265&
    limit=300000&
    timeResolution=hour&
    parameterId=mean_radiation&
    datetime=2021-01-01T00:00:00Z/2022-12-31T23:00:00Z&
    api-key=9fa4ba93-1bb9-48c8-8706-b4e6565e83e4
"""

import pandas as pd
import os, json, csv

cwd = os.getcwd()

""" Load json files from cwd """
temp_dir = os.path.join(cwd, 'mean_wind_speed.json')
with open(temp_dir) as data_file:
    wind = json.load(data_file)

temp_dir = os.path.join(cwd, 'mean_temp.json')
with open(temp_dir) as data_file:
    temp = json.load(data_file)

temp_dir = os.path.join(cwd, 'mean_pressure.json')
with open(temp_dir) as data_file:
    press = json.load(data_file)
    
""" Create dataframe with datetime observations """
time, time_check, wind_speed, temperature, pressure = [], [], [], [], []

for results in wind['features']:
    time.insert(0,results['properties']['from'])
    wind_speed.insert(0,results['properties']['value'])

for results in temp['features']:
    time_check.insert(0,results['properties']['from'])
    temperature.insert(0,results['properties']['value'])

if (not time == time_check):
    print("Time stamps for the data files do not match -")
    print("Request new data if not intended.")

time_check = []

for results in press['features']:
    time_check.insert(0,results['properties']['from'])
    pressure.insert(0,results['properties']['value'])

if (not time == time_check):
    print("Time stamps for the data files do not match -")
    print("Request new data if not intended.")

df = pd.DataFrame([time,wind_speed,temperature,pressure]).T
df.columns = 'HourUTC','wind_speed [m/s]','temperature [C]','pressure [hPa]'
df['HourUTC'] = pd.to_datetime(df['HourUTC'])
df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
df.set_index('HourUTC', inplace=True)


temp_dir = os.path.join(cwd, 'data.csv')
df.to_csv(temp_dir)

""" Print statements... """
"""
print(df)
print(koege['features'][-1])
print(koege['features'][-1]['properties']['from']) #prints last time recorded
print(koege['features'][-1]['properties']['value']) #prints last value recorded
"""


"""
plt.plot(df['2022'])
plt.ylabel('Radiation [W/m^2]')
plt.xticks(rotation=30)
plt.show()
"""
