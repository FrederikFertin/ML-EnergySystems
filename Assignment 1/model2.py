
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from createXY import prepData, loadBids
import os
import pandas as pd
from createOptBids import runOpt, revenue_calc
from model1 import *

#%% Import data
cwd = os.getcwd()

filename = os.path.join(cwd, 'Prices.csv')
prices = pd.read_csv(filename)
spot = np.array(prices['Spot'])

up = np.array(prices['Up'])
down = np.array(prices['Down'])

_, _, spot_test, up_test = train_test_split(spot, up, test_size=0.4, shuffle=False)

filename = os.path.join(cwd, 'wind power clean.csv')
power = pd.read_csv(filename)
power = np.array(power['Actual'])

_, _, down_test, power_test = train_test_split(down, power, test_size=0.4, shuffle=False)

filename = os.path.join(cwd, 'optimal bids.csv')
opt_bids = pd.read_csv(filename)


#%%


