import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from createXY import prepData, loadBids

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

def gradient_descent(X, y, M, alpha):
    theta_old = np.zeros(len(X[0,:]))
    theta_new = theta_old.copy()
    
    N = len(X[:,0])
    
    for k in range(M):
        # alpha = 1/(lambda_*(k+1))
        theta_new = theta_old - alpha * sum((np.transpose(theta_old) @ X[i] - y[i]) * X[i] for i in range(N)) 
        theta_old = theta_new
    return theta_old

def closed_form_fit(X, y):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X) @ XT @ y

def closed_form_predict(beta,X):
    return X @ beta


#%% Data set
# "model1"
# "model2"
task = "model2"

data = prepData()
bids = loadBids()
if task == "model1":
    y = np.array(data['production'])
elif task == "model2":
    y = bids

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
X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=0.4, shuffle=False)


# Solution using gradient descent 
M = 1000
alpha = 1/1000
beta_GD = gradient_descent(X_train, y_train, M, alpha)
y_pred_train_GD = closed_form_predict(beta_GD, X_train)
mse_train_GD = mean_squared_error(y_train,y_pred_train_GD)
y_pred_test_GD = closed_form_predict(beta_GD, X_test)
mse_test_GD = mean_squared_error(y_test,y_pred_test_GD)
print("Gradient descent:")
print("Model coefficients: ", beta_GD)
print('Training mse: ', mse_train_GD)
print("Test mse: ", mse_test_GD)

# Solution using closed form (i.e., matrix formula)
beta_CF = closed_form_fit(X_train,y_train)
y_pred_train_CF = closed_form_predict(beta_CF, X_train)
mse_train_CF = mean_squared_error(y_train,y_pred_train_CF)
y_pred_test_CF = closed_form_predict(beta_CF, X_test)
mse_test_CF = mean_squared_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mse: ', mse_train_CF)
print("Test mse: ", mse_test_CF)

#%% Step 3.2-3

'''
beta_CF = closed_form_fit(X_train,y_train)
y_pred_train_CF = closed_form_predict(beta_CF, X_train)
mse_train_CF = mean_squared_error(y_train,y_pred_train_CF)
y_pred_test_CF = closed_form_predict(beta_CF, X_test)
mse_test_CF = mean_squared_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mse: ', mse_train_CF)
print("Test mse: ", mse_test_CF)
'''
plotting = False
if plotting:    
    feat = [['ones', 'wind_speed [m/s]']]
    sizes = [len(data)]
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

        X_train, X_test, y_train, y_test = train_test_split(X_slice, y_slice, test_size=0.4, shuffle=False)

        # Closed form linear regression:
        beta = closed_form_fit(X_train, y_train)
        y_pred = closed_form_predict(beta, X_test)
        mse = mean_squared_error(y_test, y_pred)  # 0.028742528161411984
        mse_list.append(mse)
print(mse_list)

# Plotting
if plotting:
    plt.scatter(data['wind_speed [m/s]'],data['production'],s=1)
    plt.scatter(X_test[:,1],y_pred,s=10)
    plt.xlabel('Wind speed')
    plt.ylabel('Production')
    plt.show

#%% Chosen features for the extensions 
X = np.array(data[['ones', 'wind_speed [m/s]']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

#%% Step 4.1
X_poly_train = np.transpose(np.array([X_train[:,0],X_train[:,1],X_train[:,1]**2])) #,X_train[:,1]**3]))
X_poly_test = np.transpose(np.array([X_test[:,0],X_test[:,1],X_test[:,1]**2])) #,X_test[:,1]**3]))

beta = closed_form_fit(X_poly_train, y_train)
y_pred_train = closed_form_predict(beta, X_poly_train)
mse_train = mean_squared_error(y_train, y_pred_train)
y_pred_test = closed_form_predict(beta, X_poly_test)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Polynomial regression:")
print("Model coefficients: ", beta)
print('Training mse: ', mse_train)
print("Test mse: ", mse_test)

#%% Step 4.2
# Gaussian kernel 
def gaussian_kernel(x_t,x_u,sigma=0.5):
    # Finds the weight of a data point x_t given a fitting point x_u
    return  np.exp(-np.linalg.norm(x_t-x_u) / (2*sigma**2) )

def cf_weighted_fit(X, y, W):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ W @ X) @ XT @ W @ y

def weighted_regression_fit(X_train, y_train, M = 10, lambda_ = 0): # M is # fitting points
    # Number of data points
    N = np.shape(X_train)[0]
    
    # Quantiles 0.1,...,1.0
    quantiles = np.linspace(0,1,M) 
    
    # Sorted indices based on column 1 (Wind speeds)
    ix = np.argsort(X_train[:,1])
    X_sorted = X_train[ix]

    # Indices of the chosen quantiles
    q_ix = ((N-1)*quantiles).astype(int)
    
    # Fitting points X_u are data points corresponding to quantiles in 
    # the feature wind speeds
    X_u = X_sorted[q_ix]
    
    # Initialize y values of fitting points to zero
    y_u = np.zeros(M)
    
    # Loop through fitting points 
    for i in range(M):
        W = np.zeros((N,N))
        
        # Find weights of all data points given fitting point i
        for j in range(N):
            W[j,j] = gaussian_kernel(X_train[j],X_u[i])
        
        # Find parameters beta and y value for fitting point i
        if lambda_ > 0:
            beta = cf_regu_weighted_fit(X_train, y_train, W, lambda_)
        else:
            beta = cf_weighted_fit(X_train, y_train, W)
        y_u[i] = closed_form_predict(beta, X_u[i])
    
    return X_u, y_u

def weighted_regression_predict(X_test, X_u, y_u):
    # Number of test points
    N = np.shape(X_test)[0]
    
    # Number of fitting points
    M = np.shape(X_u)[0]
    
    y_pred = np.zeros(N)
    
    # Loop through test points
    for j in range(N):
        
        # Loop through fitting points and calculate the distances between 
        # the test point and the fitting points 
        dist = np.zeros(M)
        for i in range(M):
            dist[i] = np.linalg.norm(X_test[j] - X_u[i])
        
        # Find indices of the two nearest fitting points
        min_ixs = np.argpartition(dist, 2)[:2]
        
        # Calculate alpha to perform linear interpolation between the two
        # nearest fitting points: 
        # alpha = dist 2nd nearest / (dist nearest + dist 2nd nearest)
        alpha = dist[min_ixs[1]] / (dist[min_ixs[0]] + dist[min_ixs[1]])
        alpha = np.nan_to_num(alpha)

        # y_pred = alpha*nearest + (1-alpha)*2nd_nearest
        y_pred[j] = alpha * y_u[min_ixs[0]] + (1-alpha) * y_u[min_ixs[1]]
    
    return y_pred

X_u, y_u = weighted_regression_fit(X_train, y_train)
y_pred = weighted_regression_predict(X_test, X_u, y_u)
mse = mean_squared_error(y_test, y_pred)
print("MSE using weighted linear regression: " + str(mse))

#%% Step 5.1
############# Linear Ridge regression ###############
def cf_regu_fit(X,y,lambda_):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X + lambda_) @ XT @ y

def cf_regu_weighted_fit(X, y, W, lambda_):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ W @ X + lambda_) @ XT @ W @ y

# Regularized linear regression
beta_regu = cf_regu_fit(X_train,y_train,0.5)
y_pred_train_regu = closed_form_predict(beta_regu, X_train)
mse_train_regu = mean_squared_error(y_train,y_pred_train_regu)
y_pred_test_regu = closed_form_predict(beta_regu, X_test)
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
        
        beta = cf_regu_fit(xx_train,yy_train,l)
        y_pred = closed_form_predict(beta, xx_val)
        mse_lr[-1] += mean_squared_error(yy_val,y_pred) / K

print(mse_lr)
print()

# Find the lambda value which results in the lowest validation mse  
min_ix = np.argmin(np.asarray(mse_lr))
lambda_ = lambda_[min_ix]

# Train the best regularized model on entire training set and evaluate on 
# test set
beta = cf_regu_fit(X_train,y_train,lambda_)
y_pred = closed_form_predict(beta, X_test)
mse_lr = mean_squared_error(y_test,y_pred)
print("Best regularized linear regression:")
print("Model coefficients: ", beta)
print("Test mse: ", mse_lr)
print()

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




























