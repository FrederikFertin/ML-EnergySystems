# General packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# Local scripts and functions
from createXY import prepData, loadBids
from createOptBids import revenue_calc
from regression import *


#%% Data set
split = 0.25

data, mu_data, std_data = prepData()
y = loadBids()


#%% Prepare price data for revenue calculations
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


#%% Feature


