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
from createXY import prepData
from createOptBids import runOpt, revenue_calc
from regression import *

"""
def stochastic_gradient_descent(X, y, N, lambda_):
    theta_old = np.zeros((1,len(X[0,:])))
    theta_new = np.zeros(len(X[0,:]))
    mse_train = np.asarray([])
    mse_test = np.asarray([])
    for k in range(N):
        i = np.random.randint(0,len(y)-1)
        alpha = 1/(lambda_*(k+1))
        a = y[i]*np.dot(np.transpose(theta_old[k]),X[i,:])
        if (a < 1):
            theta_new = theta_old[k] - alpha*(lambda_*theta_old[k] - y[i]*X[i,:])
        else:
            theta_new = theta_old[k] - alpha*lambda_*theta_old[k]
        
        y_pred_train = np.dot(theta_old[-1,:],np.transpose(X_train))
        y_pred_train = np.sign(y_pred_train)
        mse_train = np.append(mse_train, mean_squared_error(y_train,y_pred_train))
        
        y_pred_test = np.dot(theta_old[-1,:],np.transpose(X_test))
        y_pred_test = np.sign(y_pred_test)
        mse_test = np.append(mse_test, mean_squared_error(y_test,y_pred_test))
        
        theta_old = np.append(theta_old,np.asarray([theta_new]),axis=0)
        
    return theta_old, mse_train, mse_test
"""

#%% Data set
split = 0.4

data, mu_data, std_data = prepData()
y = np.array(data['production'])


#%% Prepare price data for revenue calculations
cwd = os.getcwd()

filename = os.path.join(cwd, 'Prices.csv')
prices = pd.read_csv(filename)
spot = np.array(prices['Spot'])

up = np.array(prices['Up'])
down = np.array(prices['Down'])

_, spot_test, _, up_test = train_test_split(spot, up, test_size=split, shuffle=False)

filename = os.path.join(cwd, 'wind power clean.csv')
power = pd.read_csv(filename)
power = np.array(power['Actual'])

_, down_test, _, power_test = train_test_split(down, power, test_size=split, shuffle=False)


'''
#%% Dummy data sets 
X_train = np.array([[1,1],[1,2],[1,3]])
X_test = np.array([[1,0.5]])
y_train = [3,5,7]
y_test = [2]
'''

#%% Step 3.1
""" Stochastic Gradient Descent 
N = 1000
lambda_ = 1/100
beta_SGD, mse_train_SGD, mse_test_SGD = stochastic_gradient_descent(X_train, y_train, N, lambda_)
print("Stochastic gradient descent:")
print("Model coefficients: ", beta_SGD[-1,:])
print('Training mse: ', mse_train_SGD)
print("Test mse: ", mse_test_SGD)
"""
# Choosing only 100 samples and windspeed
X = np.array(data[['ones','wind_speed [m/s]']])

X_slice = X[0:100]
y_slice = y[0:100]
X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=split, shuffle=False)


# Solution using gradient descent 
M = 1000
alpha = 1/1000
beta_GD = gradient_descent(X_train, y_train, M, alpha)
y_pred_train_GD = cf_predict(beta_GD, X_train)
mse_train_GD = mean_squared_error(y_train,y_pred_train_GD)
y_pred_test_GD = cf_predict(beta_GD, X_test)
mse_test_GD = mean_squared_error(y_test,y_pred_test_GD)
print("Gradient descent:")
print("Model coefficients: ", beta_GD)
print('Training mse: ', mse_train_GD)
print("Test mse: ", mse_test_GD)

# Solution using closed form (i.e., matrix formula)
beta_CF = cf_fit(X_train,y_train)
y_pred_train_CF = cf_predict(beta_CF, X_train)
mse_train_CF = mean_squared_error(y_train,y_pred_train_CF)
y_pred_test_CF = cf_predict(beta_CF, X_test)
mse_test_CF = mean_squared_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mse: ', mse_train_CF)
print("Test mse: ", mse_test_CF)

#%% Step 3.2-3

'''
beta_CF = cf_fit(X_train,y_train)
y_pred_train_CF = cf_predict(beta_CF, X_train)
mse_train_CF = mean_squared_error(y_train,y_pred_train_CF)
y_pred_test_CF = cf_predict(beta_CF, X_test)
mse_test_CF = mean_squared_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mse: ', mse_train_CF)
print("Test mse: ", mse_test_CF)
'''

plotting = True
if plotting:    
    feat = [['ones', 'wind_speed [m/s]']]
    sizes = [10000]
else:
    sizes = [100, 1000, len(data)]
    feat = [['ones', 'wind_speed [m/s]'], ['ones', 'wind_cubed'],
            ['ones', 'wind_speed [m/s]','temperature [C]'],
            ['ones','wind_speed [m/s]', 'pressure [hPa]'],
            ['ones', 'past_prod','wind_speed [m/s]'],
            ['ones','wind_speed [m/s]','wind_direction [degress]'],
            ['ones', 'wind_speed [m/s]','temperature [C]','pressure [hPa]','past_prod']]
mse_list = []

# Looping through features and sample sizes. Output will be a list with the shape:
    # Features[0], size[0], ..., size[-1], features[1] and so on
for features in feat:
    mse_list.append(features)
    for size in sizes:
        X = np.array(data[features])
        # X = np.matrix([x1,data[cols[0]],data[cols[1]],data[cols[2]]]).T
        X_slice = X[0:size]
        y_slice = y[0:size]

        X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=split, shuffle=False)

        # Closed form linear regression:
        beta = cf_fit(X_train, y_train)
        y_pred = cf_predict(beta, X_test)
        mse = mean_squared_error(y_test, y_pred)  # 0.028742528161411984
        mse_list.append(mse)
print(mse_list)

# Plotting
if plotting:
    plt.scatter(np.array(data['wind_speed [m/s]'])*std_data[0]+mu_data[0],data['production'],s=1)
    plt.scatter(X_test[:,1]*std_data[0]+mu_data[0],y_pred,s=1)
    plt.xlabel('Mean wind speed')
    plt.ylabel('Production')
    plt.show

#%% Chosen features for the extensions 
X = np.array(data[['ones', 'wind_speed [m/s]']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

#%% Step 4.1
X_poly_train = np.transpose(np.array([X_train[:,0],X_train[:,1],X_train[:,1]**2])) #,X_train[:,1]**3]))
X_poly_test = np.transpose(np.array([X_test[:,0],X_test[:,1],X_test[:,1]**2])) #,X_test[:,1]**3]))

beta = cf_fit(X_poly_train, y_train)
y_pred_train = cf_predict(beta, X_poly_train)
mse_train = mean_squared_error(y_train, y_pred_train)
y_pred_test = cf_predict(beta, X_poly_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Polynomial regression:")
print("Model coefficients: ", beta)
print('Training mse: ', mse_train)
print("Test mse: ", mse_test)

#%% Step 4.2
# Gaussian kernel 


X_u, y_u = weighted_regression_fit(X_train, y_train)
y_pred = weighted_regression_predict(X_test, X_u, y_u)
mse = mean_squared_error(y_test, y_pred)
print("MSE using weighted linear regression: " + str(mse))

#%% Step 5.1





# Regularized linear regression
beta_regu = l2_fit(X_train,y_train,0.5)
y_pred_train_regu = cf_predict(beta_regu, X_train)
mse_train_regu = mean_squared_error(y_train,y_pred_train_regu)
y_pred_test_regu = cf_predict(beta_regu, X_test)
mse_test_regu = mean_squared_error(y_test,y_pred_test_regu)
print("Regularized linear regression:")
print("Model coefficients: ", beta_regu)
print('Training mse: ', mse_train_regu)
print("Test mse: ", mse_test_regu)
print()

# Regularized locally weighted regression 
X_u_regu, y_u_regu = weighted_regression_fit(X_train, y_train, lambda_ = 0.5)
y_pred_train_LW_regu = weighted_regression_predict(X_train, X_u, y_u)
mse_train_LW_regu = mean_squared_error(y_train, y_pred_train_LW_regu)
y_pred_test_LW_regu = weighted_regression_predict(X_test, X_u, y_u)
mse_test_LW_regu = mean_squared_error(y_test, y_pred_test_LW_regu)
print("Regularized locally weighted linear regression:")
print('Training mse: ', mse_train_LW_regu)
print("Test mse: ", mse_test_LW_regu)
print()

#%% Step 5.2-6 Linear Regression 
# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(100,300,10)

mse_lr = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_train, y_train
    
    mse_lr.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        beta = l2_fit(xx_train,yy_train,l)
        y_pred = cf_predict(beta, xx_val)
        mse_lr[-1] += mean_squared_error(yy_val,y_pred) / K

print(mse_lr)
print()

# Find the lambda value which results in the lowest validation mse  
min_ix = np.argmin(np.asarray(mse_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
beta = l2_fit(X_train,y_train,lambda_)
y_pred = cf_predict(beta, X_test)
mse_lr = mean_squared_error(y_test,y_pred)
print("Best regularized linear regression:")
print("Model coefficients: ", beta)
print("Test mse: ", mse_lr)
print()

bid, _ = runOpt(y_pred, spot_test, up_test, down_test)
revenue = revenue_calc(bid, y_test, spot_test, up_test, down_test)

#%% Step 5.2-6 Locally weighted regression

# Values of the hyperparameter to be examined
lambda_ = np.linspace(30,50,10)

mse_lw = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_train, y_train
    
    mse_lw.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        X_u, y_u = weighted_regression_fit(xx_train, yy_train, lambda_ = l)
        y_pred = weighted_regression_predict(xx_val, X_u, y_u)
        mse_lw[-1] += mean_squared_error(yy_val, y_pred) / K

print(mse_lw)        

# Find the lambda value which results in the lowest validation mse  
min_ix = np.argmin(np.asarray(mse_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
X_u, y_u = weighted_regression_fit(X_train, y_train, lambda_ = lambda_)
y_pred = weighted_regression_predict(X_test, X_u, y_u)
mse_lw = mean_squared_error(y_test, y_pred)
print("Best regularized locally weighted linear regression:")
print("Test mse: ", mse_lw)
print()




























