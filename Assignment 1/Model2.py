# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:58:57 2023

@author: julia
"""

import pandas as pd
import os, csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import gurobipy as gb
from gurobipy import GRB
from createXY import getPrices

#%%Prepare price data
cwd = os.getcwd()
f_up = os.path.join(cwd,'Up-regulation price_2021.csv')


f_ahead = os.path.join(cwd,'Day-ahead price.xlsx')


#dayahead - selecting the correct price arrea, replacing the commas
#df_ahead = pd.read_csv(r"C:\Users\julia\OneDrive\Dokumente\DTU\machine L for E\Assignment 1\Day-ahead price.csv", sep = ";")
df_ahead = pd.read_excel(f_ahead)
df_ahead = df_ahead[df_ahead['PriceArea'] == 'DK2'].reset_index(drop = True)
df_ahead = df_ahead.stack().str.replace(',', '.').unstack()

#up and down regulation - creating columns and deleting the " in the data
df_up2022 = pd.read_csv(r"C:\Users\julia\OneDrive\Dokumente\DTU\machine L for E\Assignment 1\Up-regulation price_2022.csv", sep = ",")
df_down2022 = pd.read_csv(r"C:\Users\julia\OneDrive\Dokumente\DTU\machine L for E\Assignment 1\Down-regulation price_2022.csv", sep = ",")

target_value = '"'
df_up2022 = df_up2022.stack().str.replace(target_value,'').unstack()
df_down2022 = df_down2022.stack().str.replace(target_value,'').unstack()

# Rename Columns
# Define a dictionary to map old column names to new column names
column_mappingup = {col: col.replace('"', '') for col in df_up2022.columns}
column_mappingdown = {col: col.replace('"', '') for col in df_down2022.columns}
# Rename the columns using the mapping
df_up2022.rename(columns=column_mappingup, inplace=True)
df_down2022.rename(columns=column_mappingdown, inplace=True)

#separating into 6 columns
df_up2022[['Start time UTC','End time UTC','Start time UTC+02:00','End time UTC+02:00','Up-regulating price in the Balancing energy market']]= df_up2022['Start time UTC,End time UTC,Start time UTC+02:00,End time UTC+02:00,Up-regulating price in the Balancing energy market'].str.split(",", expand=True)
df_down2022[['Start time UTC','End time UTC','Start time UTC+02:00','End time UTC+02:00','Down-regulation price in the Balancing energy market']]= df_down2022['Start time UTC,End time UTC,Start time UTC+02:00,End time UTC+02:00,Down-regulation price in the Balancing energy market'].str.split(",", expand=True)

#print(df_up2022)
#print(df_down2022)

print(df_ahead)

#%%Create parameters for the model

power = [0.290184922, 0.258890469, 0.234708393, 0.223328592, 0.172119488, 0.160739687, 0.129445235, 0.095305832,
0.100995733, 0.120910384, 0.112375533, 0.118065434, 0.110953058, 0.073968706, 0.055476529, 0.044096728,
0.038406828, 0.034139403, 0.031294452, 0.031294452, 0.029871977, 0.027027027, 0.024182077, 0.024182077] #predicted power, used as acutal power for the next day
price_ahead = df_ahead.loc[1:24,'SpotPriceEUR'].reset_index(drop = True) #price in the first hour UTC of 2022
price_up = df_up2022.loc[2:25,'Up-regulating price in the Balancing energy market'].reset_index(drop = True)
price_down = df_down2022.loc[2:25,'Down-regulation price in the Balancing energy market'].reset_index(drop = True)

print(price_ahead)
print(price_up)
print(price_down)
#%%Create optimization model
model = gb.Model("Revenue")

#create variables
time = range(24)
b = model.addVars(time, name = 'power bid')
a = model.addVars(time, vtype=GRB.BINARY, name='a')
over = model.addVars(time, name = 'overproduction')
under = model.addVars(time, name = 'underproduction')
M = 0.01

"""
# Adding constraints for over and under production 
#if power[t] > bid[t], then a = 1, otherwise a = 0
model.addConstrs((power[t] >= b[t] - M * (1 - a) 
                 for t in time), ({'name': 'M_constr1' + str(t)} for t in time))
model.addConstrs((power[t] <= b[t] + M * a 
                 for t in time), ({'name': 'M_constr2' + str(t)} for t in time))
"""

# Add indicator constraints
model.addConstrs((over[t]  <= (power[t]-b[t]) * a[t]) for t in time)
model.addConstrs((under[t] >= (b[t]-power[t]) * (1 - a[t])) for t in time)

"""
model.addConstr(((a==0) >> (over[t] == 0) for t in time),
                     ({'name': "indicator_constr1" + str(t)} for t in time))
model.addConstr(((a==0) >> (under[t] == (b[t] - power[t]) for t in time),
                     ({'name': "indicator_constr2" + str(t)} for t in time)))
model.addConstrs(((a == 1) >> ((over[t] == power[t]-b[t])  & (under[t] == 0))
                for t in time), ({'name': "indicator_constr3" + str(t)} for t in time))
"""


objective = gb.quicksum(price_ahead[t]*b[t] + price_down[t]*over[t] - price_up[t]*under[t] for t in time)
model.setObjective(objective, gb.GRB.MAXIMIZE)

model.optimize() 

optimal_variables = model.getVars()
max_revenue = model.ObjVal
print('optimal_variables:', optimal_variables)
print('max_revenue:', max_revenue)