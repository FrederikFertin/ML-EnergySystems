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

_, spot_test, _, up_test = train_test_split(spot, up, test_size=0.4, shuffle=False)

filename = os.path.join(cwd, 'wind power clean.csv')
power = pd.read_csv(filename)
power = np.array(power['Actual'])

_, down_test, _,  power_test = train_test_split(down, power, test_size=0.4, shuffle=False)


#%% Feature addition

# Prices:
diff1 = spot-up
diff2 = down-spot

# When the up-regulation price is greater than the day ahead price, 
# it is optimal to overbid as much as possible.
overbid = diff1 > 0

# When the down-regulation price is greater than the day ahead price,
# it is optimal to underbid as much as possible.
underbid = diff2 > 0

# Number of hours where both overbidding and underbidding is
# better than actual power bidding (0):
unbalanced = underbid * overbid
# Number of hours where we bid nothing:
underbids = sum(underbid)
# Number of hours where we bid everything: 
overbids = sum(overbid)
# Number of hours where we either bid everything or nothing:
extreme_bid = sum(underbid) + sum(overbid) 
""" Conclusion: 14386 hours of the 17520 (82.1%) of the hours
it is not optimal to bid the actual power production.
Profit can be maximized by correctly predicting power production
in the 3134 remaining hours - these relations are perfect for clustering
combined with different regression models in each cluster. 

If we did not know the prices when making our bid it would potentially
be very risky to make extreme bids as the up- and down-regulation 
prices can be very large. This alone could decentivize this 
bidding strategy"""


