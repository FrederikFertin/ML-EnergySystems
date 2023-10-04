import numpy as np
import gurobipy as gb
from gurobipy import GRB


#%% Step 3
def gradient_descent(X, y, M, alpha):
    """ Gradient descent M times with step size alpha """
    
    theta_old = np.zeros(len(X[0,:]))
    theta_new = theta_old.copy()
    
    N = len(X[:,0])
    
    for k in range(M):
        # alpha = 1/(lambda_*(k+1))
        theta_new = theta_old - alpha * sum((np.transpose(theta_old) @ X[i] - y[i]) * X[i] for i in range(N)) 
        theta_old = theta_new
    return theta_old


def cf_fit(X, y):
    """ Closed form predict """
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X) @ XT @ y


def cf_predict(beta,X):
    """ Closed form fit """
    return X @ beta


#%% Step 4.2
def gaussian_kernel(x_t,x_u,sigma=0.5):
    """ Finds the weight of a data point x_t given a fitting point x_u """
    return  np.exp(-np.linalg.norm(x_t-x_u) / (2*sigma**2) )


def cf_weighted_fit(X, y, W):
    """ Closed form weighted fit """
    XT = np.transpose(X)
    return np.linalg.inv(XT @ W @ X) @ XT @ W @ y


def weighted_regression_fit(X_train, y_train, M = 10, lambda_ = 0, regularization = 'L2'): # M is # fitting points
    """ Weighted regression fit """
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
            if regularization == 'L2':
                beta = l2_weighted_fit(X_train, y_train, W, lambda_)
            elif regularization == 'L1':
                beta = l1_fit(X_train, y_train, lambda_, W = W)
            else:
                print('Regularization method not recognized\
                      - doing L2 regularization with lambda = ', lambda_)
                beta = l2_weighted_fit(X_train, y_train, W, lambda_)
        else:
            beta = cf_weighted_fit(X_train, y_train, W)
        y_u[i] = cf_predict(beta, X_u[i])
    
    return X_u, y_u


def weighted_regression_predict(X_test, X_u, y_u):
    """ Weighted regression predict """
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


#%% Step 5
############# L1 - Lasso Ridge regularization ###############
def l1_fit(X,y,lambda_,W=None):
    """ Fitting beta values using L1 regularization """
    
    model = gb.Model("L1 regularization")
    
    #create variables
    T, b = np.shape(X)
    if W == None:
        W = np.eye(T)
    T, b = range(T), range(b)
    
    beta = model.addVars(b, vtype=GRB.CONTINUOUS, name = 'beta_values')
    u = model.addVars(b, vtype=GRB.CONTINUOUS, name = 'absolute_values')
    
    # Add constraints
    model.addConstrs(u[j] >= b[j] for j in b)
    model.addConstrs(u[j] >= -b[j] for j in b)
    
    objective = gb.quicksum((y[i] - W[i,i] *\
                             gb.quicksum(beta[j] * X[i,j] for j in b)\
                             )**2 for i in T) + lambda_ * gb.quicksum(u)
    
    model.setObjective(objective, gb.GRB.MINIMIZE)
    
    model.optimize()
    
    betas = np.zeros(len(b))
    
    for i in b:
        betas[i] = beta[i].x
    
    return betas


############# L2 - Ridge regression ###############
def l2_fit(X, y, lambda_):
    """ Fitting beta values using L2 regularization """
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X + lambda_) @ XT @ y


def l2_weighted_fit(X, y, W, lambda_):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ W @ X + lambda_) @ XT @ W @ y


#%% Main
if __name__ == "__main__":
    print('No main code to execute')
