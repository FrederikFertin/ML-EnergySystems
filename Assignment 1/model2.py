# General packages
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# Local scripts and functions
from createXY import prepData, loadBids
from createOptBids import revenue_calc
from regression import *
import seaborn as sns


#%% Data set
split = 0.25
# Splitting data into half train half test. Splitting test into half test half validation
# We should discuss which splits we use again
training_test_split = 0.5
test_val_split = 0.5

data, mu_data, std_data = prepData()
y = loadBids()
prod = data['production']

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
M = 0.001
# Prices:
diff1 = spot-up
diff2 = down-spot

# When the up-regulation price is greater than the day ahead price,
# it is optimal to overbid as much as possible.
overbid = diff1 > 0+M

# When the down-regulation price is greater than the day ahead price,
# it is optimal to underbid as much as possible.
underbid = diff2 > 0+M

# Number of hours where both overbidding and underbidding is
# better than actual power bidding (0):
unbalanced = underbid == overbid
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

diff1 = (diff1 - np.mean(diff1[0:int(len(diff1)*(1-split))]))/np.std(diff1[0:int(len(diff1)*(1-split))])
diff2 = (diff2 - np.mean(diff2[0:int(len(diff1)*(1-split))]))/np.std(diff2[0:int(len(diff1)*(1-split))])
spot_z = (spot - np.mean(spot[0:int(len(diff1)*(1-split))]))/np.std(spot[0:int(len(diff1)*(1-split))])

#%% Feature selecti0on
data['low_up'] = diff1
data['high_down'] = diff2
data['spot'] = spot_z
sel_features, mae_list, mae_dict = feature_selection_linear(data, y, training_test_split=training_test_split
                                                  , test_val_split=test_val_split)



'''From here on down it is the exact same as model 1.py
 - except when defining the bid before revenue calculation. '''


#%% Chosen features for the extensions
X = np.array(data[sel_features])
#X = np.array(data[['ones', 'wind_speed [m/s]']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

### Calculation of optimal daily revenue: ###
opt_rev = revenue_calc(power_test, y_test, spot_test, up_test, down_test)

#%% Linear regression model
beta = cf_fit(X_train, y_train)
y_pred_lr = cf_predict(beta, X_test)
y_pred_train = cf_predict(beta, X_train)
mae_train_lr = mean_absolute_error(y_train, y_pred_train)
mae_test_lr = mean_absolute_error(y_test, y_pred_lr)
y_pred = np.array(y_pred_lr)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 1] = 1
revenue_lr = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)

### L2 regularization CV of linear model ###
lambda_lr = cvL2(X_train, y_train, lb = 40, ub = 50) # ub and lb found through an iterative process
beta_regu = l2_fit(X_train, y_train, lambda_lr)
y_pred_l2lr = cf_predict(beta_regu, X_test)
y_pred_train = cf_predict(beta_regu, X_train)
mae_train_l2lr = mean_absolute_error(y_train, y_pred_train)
mae_test_l2lr = mean_absolute_error(y_test, y_pred_l2lr)
y_pred = np.array(y_pred_l2lr)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 1] = 1
y_bid_l2lr = np.array(y_pred)
revenue_l2lr = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)


#%% Polynomial features
data['w2'] = data['wind_speed [m/s]']**2
data['w3'] = data['wind_speed [m/s]']**3
data['t2'] = data['temperature [C]']**2
data['p2'] = data['past_prod']**2
argw1 = np.where(sel_features == 'wind_speed [m/s]')[0][0]
sel_features = np.insert(sel_features,argw1+1,'w3')
sel_features = np.insert(sel_features,argw1+1,'w2')
argt1 = np.where(sel_features == 'temperature [C]')[0][0]
sel_features = np.insert(sel_features,argt1+1,'t2')
argp1 = np.where(sel_features == 'past_prod')[0][0]
sel_features = np.insert(sel_features,argp1+1,'p2')

X_poly = np.array(data[sel_features])
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=split, shuffle=False)

#%% L1 feature selection
# Number of cross-validation iterations 
K = 5

# Values of the hyperparameter to be examined
lambda_ = np.linspace(0.02,0.03,11)

mae_lr = []

# Loop through lambda values 
for l in lambda_:
    
    # Redefine training sets
    xx_train, yy_train = X_train, y_train
    
    mae_lr.append(0)
    
    # Loop through cross-validation steps 
    for k in range(K):
        xx_train, xx_val, yy_train, yy_val = train_test_split(xx_train, yy_train, test_size=1/((K+1)-k), shuffle=False)
        
        clf = linear_model.Lasso(alpha=l,fit_intercept=False)
        clf.fit(xx_train,yy_train)
        
        y_pred = clf.predict(xx_val)
        mae_lr[-1] += mean_absolute_error(yy_val,y_pred) / K

# Find the lambda value which results in the lowest validation mae  
min_ix = np.argmin(np.asarray(mae_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
clf = linear_model.Lasso(alpha=lambda_,fit_intercept=False)
clf.fit(X_train,y_train)
beta = clf.coef_
coeffs_chosen = abs(beta) > 0

#%% Polynomial model after feature selection:
X_poly = X_poly[:,coeffs_chosen]
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=split, shuffle=False)

beta = cf_fit(X_train, y_train)
y_pred_pr = cf_predict(beta, X_test)
y_pred_train = cf_predict(beta, X_train)
mae_train_pr = mean_absolute_error(y_train, y_pred_train)
mae_test_pr = mean_absolute_error(y_test,y_pred_pr)
y_pred = np.array(y_pred_pr)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 1] = 1
revenue_pr = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)

### L2 regularization CV of polynomial model ###
lambda_pr = cvL2(X_train, y_train, lb = 70, ub = 90) # ub and lb found through an iterative process
beta_regu = l2_fit(X_train, y_train, lambda_pr)
y_pred_l2pr = cf_predict(beta_regu, X_test)
y_pred_train = cf_predict(beta_regu, X_train)
mae_train_l2pr = mean_absolute_error(y_train, y_pred_train)
mae_test_l2pr = mean_absolute_error(y_test, y_pred_l2pr)
y_pred = np.array(y_pred_l2pr)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 1] = 1
y_bid_l2pr = np.array(y_pred)
revenue_l2pr = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)

#%% Locally weighted regression
print('Initiating lw regression')
y_pred_lw = weighted_regression_fit_predict2(X_train, y_train, X_test, lambda_ = 0, regu = '', sigma = 0.51, message=False)
#y_pred_train = weighted_regression_fit_predict2(X_train, y_train, X_train, lambda_ = 0, regu = '', sigma = 0.51, message=False)
#mae_train_lw = mean_absolute_error(y_train, y_pred_train)
mae_test_lw = mean_absolute_error(y_test, y_pred_lw)
y_pred = np.array(y_pred_lw)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 1] = 1
revenue_lw = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)

#%% Locally weighted regression - regularized
### L2 regularization ###
# Lambda has been determined using cross validation (lambda_lw = 1)
print('Initiating regu. lw regression')
lambda_lw = 1
y_pred_l2lw = weighted_regression_fit_predict2(X_train, y_train, X_test, lambda_ = lambda_lw, regu = 'L2', sigma = 0.51, message=True)
mae_test_l2lw = mean_absolute_error(y_test, y_pred_l2lw)
#y_pred_train = weighted_regression_fit_predict2(X_train, y_train, X_train, lambda_ = lambda_lw, regu = 'L2', sigma = 0.51, message=True)
#mae_train_l2lw = mean_absolute_error(y_train, y_pred_train)
y_pred = np.array(y_pred_l2lw)
y_pred[y_pred_l2lw < 0] = 0
y_pred[y_pred_l2lw > 1] = 1
y_bid_l2lw = np.array(y_pred)
revenue_l2lw = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)

#%% Clustering of bid predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

def clusterBid(X_train, y_train, X_test, ix1 = 1, ix2 = 4):
    # ix1 = 1 and ix2 = 4 if it is the linear regression X
    # ix1 = 1 and ix2 = 5 if it is the polynomial regression X
    # Initialise bids at 0.5 for all hours
    y_pred = np.ones(len(X_test))*0.5
    labels = pd.Series(np.arange(len(X_test)))
    # If the down regulation price is higher than the spot price bid nothing:
    high_down = X_test[:,ix1] >= 0
    y_pred[high_down] = 0
    labels[high_down] = ['High down-regulation price']*sum(high_down)
    # If the up regulation price is lower than the spot price bid everything:
    low_up = X_test[:,ix2] >= 0
    y_pred[low_up] = 1
    labels[low_up] = ['Low up-regulation price']*sum(low_up)
    
    rest = high_down+low_up == 0
    
    # Train_rest added just before delivery and mentioned in report
    train_rest = (0 == ((X_train[:,ix1] >= 0) + (X_train[:,ix2] >= 0)) ) 
    
    y_pred[rest] = weighted_regression_fit_predict2(X_train[train_rest,:], y_train[train_rest], X_test[rest,:], lambda_ = 1, regu = 'L2', sigma = 0.51, message=False)
    labels[rest] = ['Other market conditions']*sum(rest)
    
    return y_pred, labels

y_pred_cl, labels_test = clusterBid(X_train, y_train, X_test)
mae_test_cluster = mean_absolute_error(y_test, y_pred)
y_pred_train, labels_train = clusterBid(X_train, y_train, X_train)
mae_train_cluster = mean_absolute_error(y_train, y_pred_train)
y_pred = np.array(y_pred_cl)
y_pred[y_pred < 0] = 0
y_pred[y_pred > 1] = 1
y_bid_cl = np.array(y_pred)
revenue_cluster = revenue_calc(power_test, y_pred, spot_test, up_test, down_test)

#%% Visualizations
plt.rcParams.update({'font.size': 13})

df_preds = pd.DataFrame()
df_preds['Opt preds'] = y_test
df_preds['LR preds'] = y_pred_l2lr
df_preds['PR preds'] = y_pred_l2pr
df_preds['LW preds'] = y_pred_l2lw
df_preds['CL preds'] = y_pred_cl

plt.figure(figsize=(6,6))
sns.set(font_scale=0.8)
sns.histplot(df_preds,fill=False,palette=sns.color_palette("tab10")[0:5],linewidth = 1.5).set(title='Model predictions')
plt.xlabel('Predicted optimal bid')
plt.xlim(-1,2)
#plt.legend(title='Model', fontsize=10)
plt.show()

sns.displot(df_preds, kind='kde', bw_adjust=0.5)

df_bids = pd.DataFrame()
df_bids['Opt bids'] = y_test
df_bids['LR bids'] = y_bid_l2lr
df_bids['PR bids'] = y_bid_l2pr
df_bids['LW bids'] = y_bid_l2lw
df_bids['CL bids'] = y_bid_cl

plt.figure(figsize=(6,6))
sns.set(font_scale=0.8)
sns.histplot(df_bids,fill=False,palette=sns.color_palette("tab10")[0:5],linewidth = 1.5).set(title='Model bids')
plt.xlabel('Bids')
#plt.xlim(-1,2)
#plt.legend(title='Model', fontsize=10)
plt.show()


sns.histplot(df_bids, fill=False)
sns.displot(df_bids, kind='kde', bw_adjust=.5)

df_cl = pd.DataFrame()
df_cl['Market balance'] = labels_test
df_cl['Predictions'] = y_pred_cl
df_cl['Bids'] = y_bid_cl
sns.histplot(df_cl, x = 'Predictions', hue='Market balance')
plt.show()

colors = sns.color_palette()
#plt.hist(y_pred_cl,color=colors[0],alpha=0.5,label='Cluster model')
plt.hist(y_pred_l2lw,color=colors[1],alpha=0.5,label='L2 LW. model')
#plt.hist(y_pred_l2lw_actual,color=colors[2],alpha=0.5,label='L2 LW. model - actual bids')
plt.hist(y_pred_l2lr,color=colors[3],alpha=0.5,label='L2 LR model')
plt.hist(y_pred_l2pr,color=colors[4],alpha=0.5,label='L2 PR model')
plt.xlabel('Model Bids')
plt.ylabel('Occurences')
plt.legend(loc='upper center')
plt.show()

