import numpy as np
from sklearn.metrics import mean_squared_error


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
    # theta_old = np.zeros((1,len(X[0,:])))
    # theta_new = np.zeros(len(X[0,:]))
    theta_old = np.zeros(len(X[0,:]))
    theta_new = theta_old.copy()
    
    N = len(X[:,0])
    
    for k in range(M):
        # alpha = 1/(lambda_*(k+1))
        theta_new = theta_old - alpha * sum((np.transpose(theta_old) @ X[i] - y[i]) * X[i] for i in range(N)) 
        theta_old = theta_new
    return theta_old

def closed_form_solve(X, y):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X) @ XT @ y

def closed_form_predict(beta,X):
    return X @ beta

def closed_form_weighted_solve():
    
    
#%% Dummy data sets 
X_train = np.array([[1,1],[1,2],[1,3]])
X_test = np.array([[1,0.5]])
y_train = [3,5,7]
y_test = [2]

M = 100000
alpha = 1/100000
# lambda_ = 1/101


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
beta_CF = closed_form_solve(X_train,y_train)
y_pred_train_CF = closed_form_predict(beta_CF, X_train)
mse_train_CF = mean_squared_error(y_train,y_pred_train_CF)
y_pred_test_CF = closed_form_predict(beta_CF, X_test)
mse_test_CF = mean_squared_error(y_test,y_pred_test_CF)
print("Closed form:")
print("Model coefficients: ", beta_CF)
print('Training mse: ', mse_train_CF)
print("Test mse: ", mse_test_CF)





































