import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

def create_train_test(df, train_days, len_of_train = 7, len_of_test = 1, d=0):
    df_train = df.iloc[:(train_days + d) * 24, :]
    p_train = df_train['Spot']
    # Train only on one week of price data
    p_train.iloc[len(p_train)-len_of_train*24:]
    
    # Test only on the next 24 hours
    df_test = df.iloc[(train_days + d) * 24:(train_days + d + len_of_test) * 24, :]
    p_test = df_test['Spot']
    
    return p_train, p_test

def create_second_train_test(prices, train_days, test_days):
    
    train_end = 8760+train_days*24
    test_end = train_end + test_days*24
    
    p_train2 = prices[8760:train_end]
    p_test2 = prices[train_end:test_end]
    
    return p_train2, p_test2
    

def cf_fit(X, y):
    """ Closed form predict """
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X) @ XT @ y


def cf_predict(beta, X):
    """ Closed form fit """
    return X @ beta


def getPriceLevels(p_train, n_levels):
    """
    Computes the quantiles of the price data set given
    and returns the prices changed to the mean of the price level they are in.
    
    Args: 
        p_train: prices from training data (pd.Series)
        n_levels: number of desired price levels/quantiles
    
    Returns:
            prices: Prices changed to the mean of each quantile.
            price_levels: The price level of each quantile
            price_cuts: The actual quantile values, which the prices have been divided by.
    """
    
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
    """
    Determines the optimal bidding strategy with perfect knowledge during 
    the given testing period. 
    
    Args: 
        p_test: pd.DataFrame containing price data for the testing period.
    
    Returns: 
        model.objVal: total profits during testing period.
        SOC: list of SOC during the testing period.
        p_ch: list of (dis)charging rates during testing period. 
    """
    
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

def continual_test(df, model, gamma = None, maxIter = 100, train_days = 180, test_days = 50, length_train = 7, length_test = 1, n_levels = 10):
    profits = np.array([0])
    socs = np.array([200])
    if model.model_type == "discrete":
        if gamma == None:
            gamma = 1
        for d in range(0,test_days,length_test):
            print(d/length_test+1 ,"/", test_days/length_test)
            p_train, p_test = create_train_test(df, train_days, len_of_train = length_train, len_of_test = length_test, d=d)
            p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_levels)
            model.setPriceLevels(p_levels, p_cuts)
            model.calcPmatrix(p_trains)
            model.valueIter(gamma = gamma, maxIter = maxIter)
            profit, soc = model.test(p_test, soc = socs[-1], profit = profits[-1])
            profits = np.insert(profits,len(profits),profit)
            socs = np.insert(socs,len(socs),soc)
        return profits, socs
    elif model.model_type == "continuous":
        if gamma == None:
            gamma = 0.952
        for d in range(0,test_days,length_test):
            print(d/length_test+1 ,"/", test_days/length_test)
            p_train, p_test = create_train_test(df, train_days, len_of_train = length_train, len_of_test = length_test, d=d)
            if model.sampling:
                p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_levels)
                model.calc_k_samplers(p_train,p_levels, p_trains)
                model.fitted_value_iteration_k_sampling(p_train, maxIter=maxIter, gamma=gamma)
            else:
                model.fitted_value_iteration(p_train, nSamples = 200, maxIter=maxIter, gamma = gamma)
            profit, soc = model.test(p_test, soc = socs[-1], profit = profits[-1])
            profits = np.insert(profits,len(profits),profit)
            socs = np.insert(socs,len(socs),soc)
        return profits, socs
    else:
        print("Unknown Model")












