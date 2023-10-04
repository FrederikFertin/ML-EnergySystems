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

filename = os.path.join(cwd, 'Prices.csv')
prices = pd.read_csv(filename)
spot = np.array(prices['Spot'])
up = np.array(prices['Up'])
down = np.array(prices['Down'])

filename = os.path.join(cwd, 'wind power clean.csv')
power = pd.read_csv(filename)
power = np.array(power['Actual'])


#%%Create optimization model
model = gb.Model("Revenue")

#create variables
time = range(len(spot))
b = model.addVars(time, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = 'power_bid')
over = model.addVars(time, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = 'overproduction')
under = model.addVars(time, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = 'underproduction')
a = model.addVars(time, vtype=GRB.BINARY, name='a')

# Add constraints
model.addConstrs((over[t]  <= (power[t]-b[t]) * a[t]) for t in time)
model.addConstrs((under[t] >= (b[t]-power[t]) * (1 - a[t])) for t in time)


objective = gb.quicksum(spot[t]*b[t] + down[t]*over[t] - up[t]*under[t] for t in time)
model.setObjective(objective, gb.GRB.MAXIMIZE)

model.optimize()

optimal_variables = model.getVars()
max_revenue = model.ObjVal

print('max_revenue:', max_revenue)

#%% Save optimal bids
bids = np.zeros(len(b))

for i in range(len(b)):
    bids[i] = b[i].x

df_bid = pd.DataFrame()
df_bid['Opt-Bid'] = bids

filename = os.path.join(cwd,'optimal bids.csv')
df_bid.to_csv(filename)
