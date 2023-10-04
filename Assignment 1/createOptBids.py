import pandas as pd
import os
import numpy as np
import gurobipy as gb
from gurobipy import GRB
# import time as time

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
# start1 = time.time()

model = gb.Model("Revenue")

#create variables
T = range(len(spot))
b = model.addVars(T, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = 'power_bid')
over = model.addVars(T, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = 'overproduction')
under = model.addVars(T, vtype=GRB.CONTINUOUS, lb = 0, ub = 1, name = 'underproduction')
a = model.addVars(T, vtype=GRB.BINARY, name='a')

# Add constraints
#model.addConstrs((over[t]  <= (power[t]-b[t]) * a[t]) for t in T)
#model.addConstrs((under[t] >= (b[t]-power[t]) * (1 - a[t])) for t in T)
# Faster approach:
model.addConstrs((over[t]  == (power[t]-b[t]) * a[t]) for t in T)
model.addConstrs((under[t] == (b[t]-power[t]) * (1 - a[t])) for t in T)

objective = gb.quicksum(spot[t]*b[t] + down[t]*over[t] - up[t]*under[t] for t in T)
model.setObjective(objective, gb.GRB.MAXIMIZE)

model.optimize()

optimal_variables = model.getVars()
max_revenue = model.ObjVal

print('max_revenue:', max_revenue)

# end1 = time.time()
#%% Save optimal bids
bids = np.zeros(len(b))

for i in range(len(b)):
    bids[i] = b[i].x

df_bid = pd.DataFrame()
df_bid['Opt-Bid'] = bids

#filename = os.path.join(cwd,'optimal bids.csv')
#df_bid.to_csv(filename)
