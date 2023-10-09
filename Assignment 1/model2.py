# General packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# Local scripts and functions
from createXY import prepData, loadBids
from createOptBids import revenue_calc
from regression import *


#%% Data set
split = 0.25
# Splitting data into half train half test. Splitting test into half test half validation
# We should discuss which splits we use again
training_test_split = 0.5
test_val_split = 0.5

data, mu_data, std_data = prepData()
y = loadBids()


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

_, down_test, _,  power_test = train_test_split(down, power, test_size=split, shuffle=False)


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

#%%
data['overbid'] = diff1
#data['underbid'] = diff2
#data['spot'] = spot
sel_features, mse_list, mse_dict = feature_selection_linear(data, y, training_test_split=training_test_split
                                                  , test_val_split=test_val_split)



'''From here on down it is the exact same as model 1.py
 - except when defining the bid before revenue calculation. '''


#%% Chosen features for the extensions
X = np.array(data[sel_features])
#X = np.array(data[['ones', 'wind_speed [m/s]']])

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

bid = y_pred
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

# Calculating MSE
mse_cluster = mean_squared_error(y_test, y_pred_cluster_sort)


for i in range(n_clusters):
    plt.scatter(X_train_sets[i][:,1], y_train_sets[i], s=1)

# Done in separate loop to ensure that predicted values are on top
for i in range(n_clusters):
    plt.scatter(X_test_sets[i][:,1],y_pred_cluster[i][:,0],s=1)
    
plt.show()
        
        
    
    














