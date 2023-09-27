import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
    
#%% Dummy data sets 
X_train = np.array([[1,1],[1,2],[1,3]])
X_test = np.array([[1,0.5]])
y_train = [3,5,7]
y_test = [2]


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
# Solution using gradient descent 
M = 100000
alpha = 1/100000
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
beta_CF = closed_form_fit(X_train,y_train)
y_pred_train_CF = closed_form_predict(beta_CF, X_train)
mse_train_CF = mean_squared_error(y_train,y_pred_train_CF)
y_pred_test_CF = closed_form_predict(beta_CF, X_test)
mse_test_CF = mean_squared_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mse: ', mse_train_CF)
print("Test mse: ", mse_test_CF)

#%% Step 4.1
poly = PolynomialFeatures(degree=2, include_bias=False)

poly_features_train = poly.fit_transform(X_train.reshape(-1, 1))

poly_features_test = poly.fit_transform(X_test.reshape(-1, 1))

poly_reg_model = LinearRegression()

poly_reg_model.fit(poly_features_train, y_train)

y_pred = poly_reg_model.predict(poly_features_test)

mse = mean_squared_error(y_test, y_pred)

#%% Step 4.2
# Gaussian kernel 
def gaussian_kernel(x_t,x_u,sigma=0.05):
    return  np.exp(-np.linalg.norm(x_t-x_u) / (2*sigma**2) )

def closed_form_weighted_fit(X, y, W):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ W @ X) @ XT @ W @ y

N = np.shape(X_train)[0]

quantiles = np.linspace(0,1,11) 
ix = np.argsort(X_train[:,1])
X_sorted = X_train[ix]
q_ix = ((len(X_sorted[0])-1)*quantiles).astype(int)
X_u = X_sorted[[q_ix]]
y_u = np.zeros(len(X_u[0]))
for i in range(len(X_u[0])):
    W = np.zeros((N,N))
    for j in range(len(X_train[0])):
        W[j,j] = gaussian_kernel(X_train[j],X_u[i])
    beta = closed_form_weighted_fit(X_train, y_train, W)
    y_u[i] = closed_form_predict(beta, X_u[i])


#%% Step 5.1
def cf_reg_fit(X,y,lambda_):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X + lambda_) @ XT @ y






































