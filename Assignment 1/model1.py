# General packages
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import pandas as pd

# Scikit-learn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
from sklearn import linear_model

# Local scripts and functions
from createXY import prepData
from createOptBids import runOpt, revenue_calc
from regression import *

"""
def stochastic_gradient_descent(X, y, N, lambda_):
    theta_old = np.zeros((1,len(X[0,:])))
    theta_new = np.zeros(len(X[0,:]))
    mae_train = np.asarray([])
    mae_test = np.asarray([])
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
        mae_train = np.append(mae_train, mean_absolute_error(y_train,y_pred_train))
        
        y_pred_test = np.dot(theta_old[-1,:],np.transpose(X_test))
        y_pred_test = np.sign(y_pred_test)
        mae_test = np.append(mae_test, mean_absolute_error(y_test,y_pred_test))
        
        theta_old = np.append(theta_old,np.asarray([theta_new]),axis=0)
        
    return theta_old, mae_train, mae_test
"""

#%% Data set
split = 0.25

# Splitting data into half train half test. Splitting test into half test half validation
training_test_split = 0.5
test_val_split = 0.5

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



#%% Step 3.1
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
mae_train_GD = mean_absolute_error(y_train,y_pred_train_GD)
y_pred_test_GD = cf_predict(beta_GD, X_test)
mae_test_GD = mean_absolute_error(y_test,y_pred_test_GD)
print("Gradient descent:")
print("Model coefficients: ", beta_GD)
print('Training mae: ', mae_train_GD)
print("Test mae: ", mae_test_GD)

# Solution using closed form (i.e., matrix formula)
beta_CF = cf_fit(X_train,y_train)
y_pred_train_CF = cf_predict(beta_CF, X_train)
mae_train_CF = mean_absolute_error(y_train,y_pred_train_CF)
y_pred_test_CF = cf_predict(beta_CF, X_test)
mae_test_CF = mean_absolute_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mae: ', mae_train_CF)
print("Test mae: ", mae_test_CF)

#%% Step 3.2-3
plotting = False
if plotting:    
    feat = [['ones', 'wind_speed [m/s]']]
    sizes = [len(data)]


sel_features, mae_list, mae_dict = feature_selection_linear(data, y, training_test_split=training_test_split
                                                  , test_val_split=test_val_split)

# Predicting using the chosen features
y = np.array(data['production'])
X = np.array(data[sel_features])
#X = np.array(data[['ones','wind_speed [m/s]']])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=training_test_split, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=test_val_split, shuffle=False)

beta = cf_fit(X_train, y_train)
y_pred = cf_predict(beta, X_test)

MSE = mean_squared_error(y_test, y_pred)
RMSE = mean_squared_error(y_test, y_pred, squared=False)
R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)

# Plotting
if plotting:
    plt.scatter(np.array(data['wind_speed [m/s]'])*std_data[0]+mu_data[0],data['production'],s=1)
    plt.scatter(X_test[:,1]*std_data[0]+mu_data[0],y_pred,s=1)
    plt.xlabel('Mean wind speed')
    plt.ylabel('Production')
    plt.show

#%% Chosen features for the extensions
X = np.array(data[sel_features])
#X = np.array(data[['ones', 'wind_speed [m/s]']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

#%% Step 4.1 
X_poly_train = np.transpose(np.array([X_train[:,0],X_train[:,1],X_train[:,1]**2])) #,X_train[:,1]**3]))
X_poly_test = np.transpose(np.array([X_test[:,0],X_test[:,1],X_test[:,1]**2])) #,X_test[:,1]**3]))

beta = cf_fit(X_poly_train, y_train)
y_pred_train = cf_predict(beta, X_poly_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
y_pred_test = cf_predict(beta, X_poly_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
print("Polynomial regression:")
print("Model coefficients: ", beta)
print('Training mae: ', mae_train)
print("Test mae: ", mae_test)


#%% Step 4.2 vol 1

X_u, y_u = weighted_regression_fit(X_train, y_train)
y_pred = weighted_regression_predict(X_test, X_u, y_u)
mae = mean_absolute_error(y_test, y_pred)
print("mae using weighted linear regression: " + str(mae))

#%% Step 4.2 vol 2

y_pred = weighted_regression_predict2(X_train, y_train, X_test[:10])
mae = mean_absolute_error(y_test[:10], y_pred)
print("mae using weighted linear regression: " + str(mae))


#%% Step 5.1

# # Regularized linear regression
# beta_regu = l2_fit(X_train,y_train,0.5)
# y_pred_train_regu = cf_predict(beta_regu, X_train)
# mae_train_regu = mean_absolute_error(y_train,y_pred_train_regu)
# y_pred_test_regu = cf_predict(beta_regu, X_test)
# mae_test_regu = mean_absolute_error(y_test,y_pred_test_regu)
# print("Regularized linear regression:")
# print("Model coefficients: ", beta_regu)
# print('Training mae: ', mae_train_regu)
# print("Test mae: ", mae_test_regu)
# print()

# # Regularized locally weighted regression
# X_u_regu, y_u_regu = weighted_regression_fit(X_train, y_train, lambda_ = 0.5)
# y_pred_train_LW_regu = weighted_regression_predict(X_train, X_u, y_u)
# mae_train_LW_regu = mean_absolute_error(y_train, y_pred_train_LW_regu)
# y_pred_test_LW_regu = weighted_regression_predict(X_test, X_u, y_u)
# mae_test_LW_regu = mean_absolute_error(y_test, y_pred_test_LW_regu)
# print("Regularized locally weighted linear regression:")
# print('Training mae: ', mae_train_LW_regu)
# print("Test mae: ", mae_test_LW_regu)
# print()

#%% Step 5.1-6 Linear Regression

# L1 regularized linear regression
beta_regu = l1_fit(X_train,y_train,0.5)
y_pred_train_regu = cf_predict(beta_regu, X_train)
mae_train_regu = mean_absolute_error(y_train,y_pred_train_regu)
y_pred_test_regu = cf_predict(beta_regu, X_test)
mae_test_regu = mean_absolute_error(y_test,y_pred_test_regu)
print("L1 regularized linear regression:")
print("Model coefficients: ", beta_regu)
print('Training mae: ', mae_train_regu)
print("Test mae: ", mae_test_regu)
print()

# L2 regularized linear regression
beta_regu = l2_fit(X_train,y_train,0.5)
y_pred_train_regu = cf_predict(beta_regu, X_train)
mae_train_regu = mean_absolute_error(y_train,y_pred_train_regu)
y_pred_test_regu = cf_predict(beta_regu, X_test)
mae_test_regu = mean_absolute_error(y_test,y_pred_test_regu)
print("L2 regularized linear regression:")
print("Model coefficients: ", beta_regu)
print('Training mae: ', mae_train_regu)
print("Test mae: ", mae_test_regu)
print()

########## Hyperparameter selection through cross-validation ###########

########## L1 regression ##########

# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(0.005,0.006,10)

mae_lr = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_train, y_train
    
    mae_lr.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        clf = linear_model.Lasso(alpha=l)
        clf.fit(xx_train,yy_train)
        
        y_pred = clf.predict(xx_val)
        mae_lr[-1] += mean_absolute_error(yy_val,y_pred) / K

print(mae_lr)
print()

# Find the lambda value which results in the lowest validation mae
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
clf = linear_model.Lasso(alpha=lambda_)
clf.fit(X_train,y_train)
beta = clf.intercept_
beta = np.append(beta, clf.coef_)
y_pred = clf.predict(X_test)
mae_lr = mean_absolute_error(y_test,y_pred)
print("Best L1 regularized linear regression:")
print("Model coefficients: ", beta)
print("Test mae: ", mae_lr)
print()


########## L2 regression ##########

# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(100,300,10)

mae_lr = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_train, y_train
    
    mae_lr.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        beta = l2_fit(xx_train,yy_train,l)
        y_pred = cf_predict(beta, xx_val)
        mae_lr[-1] += mean_absolute_error(yy_val,y_pred) / K

print(mae_lr)
print()

# Find the lambda value which results in the lowest validation mae
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
beta = l2_fit(X_train,y_train,lambda_)
y_pred_train = cf_predict(beta, X_train)
mae_lr_train = mean_absolute_error(y_train,y_pred_train)
y_pred_test = cf_predict(beta, X_test)
mae_lr_test = mean_absolute_error(y_test,y_pred_test)
print("Best L2 regularized linear regression:")
print("Model coefficients: ", beta)
print("Test mae: ", mae_lr)
print()

bid, _ = runOpt(y_pred, spot_test, up_test, down_test)
revenue = revenue_calc(bid, y_test, spot_test, up_test, down_test)

#%% Step 5.1-6 Polynomial regression

# L1 regularized polynomial regression
beta_regu = l1_fit(X_poly_train,y_train,0.5)
y_pred_train_regu = cf_predict(beta_regu, X_poly_train)
mae_train_regu = mean_absolute_error(y_train,y_pred_train_regu)
y_pred_test_regu = cf_predict(beta_regu, X_poly_test)
mae_test_regu = mean_absolute_error(y_test,y_pred_test_regu)
print("L1 regularized polynomial regression:")
print("Model coefficients: ", beta_regu)
print('Training mae: ', mae_train_regu)
print("Test mae: ", mae_test_regu)
print()

# L2 regularized polynomial regression
beta_regu = l2_fit(X_poly_train,y_train,0.5)
y_pred_train_regu = cf_predict(beta_regu, X_poly_train)
mae_train_regu = mean_absolute_error(y_train,y_pred_train_regu)
y_pred_test_regu = cf_predict(beta_regu, X_poly_test)
mae_test_regu = mean_absolute_error(y_test,y_pred_test_regu)
print("L2 regularized polynomial regression:")
print("Model coefficients: ", beta_regu)
print('Training mae: ', mae_train_regu)
print("Test mae: ", mae_test_regu)
print()

########## Hyperparameter selection through cross-validation ###########

########## L1 regression ##########

# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(0.005,0.006,10)

mae_lr = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_poly_train, y_train
    
    mae_lr.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        clf = linear_model.Lasso(alpha=l)
        clf.fit(xx_train,yy_train)
        
        y_pred = clf.predict(xx_val)
        mae_lr[-1] += mean_absolute_error(yy_val,y_pred) / K

print(mae_lr)
print()

# Find the lambda value which results in the lowest validation mae
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
clf = linear_model.Lasso(alpha=lambda_)
clf.fit(X_poly_train,y_train)
beta = clf.intercept_
beta = np.append(beta, clf.coef_)
y_pred = clf.predict(X_poly_test)
mae_lr = mean_absolute_error(y_test,y_pred)
print("Best L1 regularized polynomial regression:")
print("Model coefficients: ", beta)
print("Test mae: ", mae_lr)
print()


########## L2 regression ##########

# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(100,300,10)

mae_lr = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_poly_train, y_train
    
    mae_lr.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        beta = l2_fit(xx_train,yy_train,l)
        y_pred = cf_predict(beta, xx_val)
        mae_lr[-1] += mean_absolute_error(yy_val,y_pred) / K

print(mae_lr)
print()

# Find the lambda value which results in the lowest validation mae
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
beta = l2_fit(X_poly_train,y_train,lambda_)
y_pred = cf_predict(beta, X_poly_test)
mae_lr = mean_absolute_error(y_test,y_pred)
print("Best L2 regularized polynomial regression:")
print("Model coefficients: ", beta)
print("Test mae: ", mae_lr)
print()

bid, _ = runOpt(y_pred, spot_test, up_test, down_test)
revenue = revenue_calc(bid, y_test, spot_test, up_test, down_test)

#%% Step 5.1-6 Locally weighted regression

# L1 Regularized locally weighted regression 
y_pred_train_LW_regu = weighted_regression_predict2(X_train, y_train, X_train, lambda_ = 0.5, regu = 'L1')
mae_train_LW_regu = mean_absolute_error(y_train, y_pred_train_LW_regu)
y_pred_test_LW_regu = weighted_regression_predict2(X_train, y_train, X_test, lambda_ = 0.5, regu = 'L1')
mae_test_LW_regu = mean_absolute_error(y_test, y_pred_test_LW_regu)

# X_u_regu, y_u_regu = weighted_regression_fit(X_train, y_train, lambda_ = 0.5)
# y_pred_train_LW_regu = weighted_regression_predict(X_train, X_u, y_u)
# mae_train_LW_regu = mean_absolute_error(y_train, y_pred_train_LW_regu)
# y_pred_test_LW_regu = weighted_regression_predict(X_test, X_u, y_u)
# mae_test_LW_regu = mean_absolute_error(y_test, y_pred_test_LW_regu)
print("L1 Regularized locally weighted linear regression:")
print('Training mae: ', mae_train_LW_regu)
print("Test mae: ", mae_test_LW_regu)
print()


# L2 Regularized locally weighted regression 
y_pred_train_LW_regu = weighted_regression_predict2(X_train, y_train, X_train, lambda_ = 0.5, regu = 'L2')
mae_train_LW_regu = mean_absolute_error(y_train, y_pred_train_LW_regu)
y_pred_test_LW_regu = weighted_regression_predict2(X_train, y_train, X_test, lambda_ = 0.5, regu = 'L2')
mae_test_LW_regu = mean_absolute_error(y_test, y_pred_test_LW_regu)

# X_u_regu, y_u_regu = weighted_regression_fit(X_train, y_train, lambda_ = 0.5)
# y_pred_train_LW_regu = weighted_regression_predict(X_train, X_u, y_u)
# mae_train_LW_regu = mean_absolute_error(y_train, y_pred_train_LW_regu)
# y_pred_test_LW_regu = weighted_regression_predict(X_test, X_u, y_u)
# mae_test_LW_regu = mean_absolute_error(y_test, y_pred_test_LW_regu)
print("L2 Regularized locally weighted linear regression:")
print('Training mae: ', mae_train_LW_regu)
print("Test mae: ", mae_test_LW_regu)
print()


########## L1 regression ##########

# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(0.001,0.01,10)

mae_lw = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_train, y_train
    
    mae_lw.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        y_pred = weighted_regression_predict2(xx_train, yy_train, xx_val, lambda_ = l, regu = 'L1')
        mae_lw[-1] += mean_absolute_error(yy_val, y_pred) / K
        
        # With vol 1
        # X_u, y_u = weighted_regression_fit(xx_train, yy_train, lambda_ = l)
        # y_pred = weighted_regression_predict(xx_val, X_u, y_u)
        # mae_lw[-1] += mean_absolute_error(yy_val, y_pred) / K

print(mae_lw)

# Find the lambda value which results in the lowest validation mae
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set

y_pred_train = weighted_regression_predict2(X_train, y_train, X_train, lambda_ = lambda_, regu = 'L1')
mae_train = mean_absolute_error(y_train, y_pred_train)
y_pred_test = weighted_regression_predict2(X_train, y_train, X_test, lambda_ = lambda_, regu = 'L1')
mae_test = mean_absolute_error(y_test, y_pred_test)

# X_u, y_u = weighted_regression_fit(X_train, y_train, lambda_ = lambda_)
# y_pred = weighted_regression_predict(X_test, X_u, y_u)
# mae_lw = mean_absolute_error(y_test, y_pred)
print("Best regularized locally weighted linear regression:")
print("Best lambda: ", lambda_)
print("Training mae: ", mae_train)
print("Test mae: ", mae_test)
print()


########## L2 regression ##########

# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(30,50,10)

mae_lw = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets 
    xx_train, yy_train = X_train, y_train
    
    mae_lw.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        y_pred = weighted_regression_predict2(xx_train, yy_train, xx_val, lambda_ = l)
        mae_lw[-1] += mean_absolute_error(yy_val, y_pred) / K
        
        # With vol 1
        # X_u, y_u = weighted_regression_fit(xx_train, yy_train, lambda_ = l)
        # y_pred = weighted_regression_predict(xx_val, X_u, y_u)
        # mae_lw[-1] += mean_absolute_error(yy_val, y_pred) / K

print(mae_lw)

# Find the lambda value which results in the lowest validation mae
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
# X_u, y_u = weighted_regression_fit(X_train, y_train, lambda_ = lambda_)

y_pred = weighted_regression_predict2(X_train, y_train, X_test, lambda_ = lambda_, regu = 'L2')
mae_lw = mean_absolute_error(y_test, y_pred)
print("Best regularized locally weighted linear regression:")
print("Test mae: ", mae_lw)
print()


# %% Step 7.1
n_clusters = 2

kmeans = KMeans(n_clusters=n_clusters,
                n_init=10,
                random_state=1)
kmeans.fit(X_train)
center_locations = kmeans.cluster_centers_
train_labels = kmeans.labels_
test_labels = kmeans.predict(X_test)
# Adding predicted labels to X_train
X_train_cluster = np.append(X_train, np.reshape(train_labels, (len(train_labels), 1)), axis=1)
# Adding temporary index labels to merge sets later
indexes = np.reshape(range(len(X_test)),(len(X_test), 1))
X_test_cluster = np.append(X_test, indexes, axis=1)
# Adding predicted labels to X_test, y_train and y_test
X_test_cluster = np.append(X_test_cluster, np.reshape(test_labels, (len(test_labels), 1)), axis=1)
y_train_cluster = np.vstack((y_train, train_labels)).T
y_test_cluster = np.vstack((y_test, test_labels)).T

# Splitting by label
X_test_sets = separate_labels(X_test_cluster, n_clusters)
X_train_sets = separate_labels(X_train_cluster, n_clusters)
y_test_sets = separate_labels(y_test_cluster, n_clusters)
y_train_sets = separate_labels(y_train_cluster, n_clusters)

# Removing and saving test indices
indices_list = []
for i in range(n_clusters):
    indices_list.append(X_test_sets[i][:,-1])
    X_test_sets[i] = np.delete(X_test_sets[i], -1,1)

beta_cluster = []
y_pred_cluster = []
# Predicting, different models can be chosen for different clusters
beta_cluster.append(cf_fit(X_train_sets[0],y_train_sets[0]))
y_pred_cluster.append(cf_predict(beta_cluster[0], X_test_sets[0]))

beta_cluster.append(cf_fit(X_train_sets[1],y_train_sets[1]))
y_pred_cluster.append(cf_predict(beta_cluster[1], X_test_sets[1]))

if n_clusters>2:
    beta_cluster.append(cf_fit(X_train_sets[2],y_train_sets[2]))
    y_pred_cluster.append(cf_predict(beta_cluster[2], X_test_sets[2]))


# Merging datapoints and sorting them to original order
for i in range(n_clusters):
    y_pred_cluster[i] = np.vstack((y_pred_cluster[i][:,0],indices_list[i])).T
y_pred_cluster_sort = np.concatenate((y_pred_cluster), axis=0)
y_pred_cluster_sort = y_pred_cluster_sort[y_pred_cluster_sort[:, -1].argsort()]
y_pred_cluster_sort = np.delete(y_pred_cluster_sort, -1,1)

# Calculating MAE
mae_cluster = mean_absolute_error(y_test, y_pred_cluster_sort)


for i in range(n_clusters):
    plt.scatter(X_train_sets[i][:,1], y_train_sets[i], s=2)


# Done in separate loop to ensure that predicted values are on top
for i in range(n_clusters):
    plt.scatter(X_test_sets[i][:,1],y_pred_cluster[i][:,0],s=1)
plt.xlabel('Mean wind speed')
plt.ylabel('Production')
plt.show()
        
        
    
    














