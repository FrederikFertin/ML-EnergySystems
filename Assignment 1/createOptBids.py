import pandas as pd
import os
import numpy as np
import gurobipy as gb
from gurobipy import GRB
# import time as time

#%%Create optimization model
# start1 = time.time()
def runOpt(y_pred, spot, up, down):
    power = y_pred
    
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
    
    max_revenue = model.ObjVal
    
    # Save optimal bids
    bids = np.zeros(len(b))
    
    for i in range(len(b)):
        bids[i] = b[i].x
    
    return bids, max_revenue


## Revenue Calculation
def revenue_calc(y_test, y_bid, ahead_rev, Up_rev, Down_rev):
   act_pow = y_test
   bid_pow = y_bid
   revenue = 0
   
   for i in range(len(bid_pow)):
      revenue += ahead_rev[i] * bid_pow[i]
      
      if (act_pow[i]-bid_pow[i])>0:
         revenue += Down_rev[i]*(act_pow[i]-bid_pow[i])
      else:
         revenue += Up_rev[i]*(act_pow[i]-bid_pow[i])
   revenue = round(revenue / len(bid_pow) *24, 2)
   return revenue


if __name__ == "__main__":
    cwd = os.getcwd()

    filename = os.path.join(cwd, 'Prices.csv')
    prices = pd.read_csv(filename)
    spot = np.array(prices['Spot'])
    up = np.array(prices['Up'])
    down = np.array(prices['Down'])
    
    filename = os.path.join(cwd, 'wind power clean.csv')
    power = pd.read_csv(filename)
    power = np.array(power['Actual'])
    
    bids, _ = runOpt(power, spot, up, down)
    df_bid = pd.DataFrame()
    df_bid['Opt-Bid'] = bids
    
    filename = os.path.join(cwd,'optimal bids.csv')
    df_bid.to_csv(filename)
