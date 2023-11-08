import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
#from sklearn.model_selection import train_test_split

def getPriceLevels(p_train,n_levels):
    prices = p_train.copy()
    p = prices.copy()
    price_cuts = np.quantile(prices, np.linspace(0,1,n_levels+1))[:n_levels]
    
    price_levels = np.zeros(n_levels)
    for i in range(n_levels):
        if i < n_levels-1:
            price_levels[i] = prices.loc[(p >= price_cuts[i]) & (p < price_cuts[i+1])].mean()
            prices.loc[(p >= price_cuts[i]) & (p < price_cuts[i+1])] = price_levels[i]
        else:
            price_levels[i] = prices.loc[p >= price_cuts[i]].mean()
            prices.loc[p >= price_cuts[i]] = price_levels[i]
    
    return prices, price_levels, price_cuts

def optimalBidding(p_test):
    T = len(p_test)
    
    model = gp.Model("Optimal Bidding") 
    
    p = model.addVars(T, lb=-100, ub=100, name='Charging rate')
    soc = model.addVars(T, lb=0, ub=500, name='SOC')
    
    model.setObjective(gp.quicksum(p[t]*p_test.iloc[t] for t in range(T)), GRB.MAXIMIZE)
    
    model.addConstr(soc[0] == 200 + p[0])
    for t in range(T-1):
        model.addConstr(soc[t+1] == soc[t] + p[t+1])
    model.addConstr(soc[T-1] == 200)
    
    model.optimize()
    
    SOC = []
    p_ch = []
    for t in range(T):
        SOC.append(soc[t].x)
        p_ch.append(p[t].x)
    
    return model.objVal, SOC, p_ch
    
#if __name__ == "__main__":

    























