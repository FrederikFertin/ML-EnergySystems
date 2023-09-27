import numpy as np
from sklearn.metric import accuracy_score

def stochastic_gradient_descent(X, y, N, lambda_):   
    theta_old = np.zeros((1,len(X[0,:])))
    theta_new = np.zeros(len(X[0,:]))
    acc_train = np.asarray([])
    acc_test = np.asarray([])
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
        acc_train = np.append(acc_train, accuracy_score(y_train,y_pred_train))
        
        y_pred_test = np.dot(theta_old[-1,:],np.transpose(X_test))
        y_pred_test = np.sign(y_pred_test)
        acc_test = np.append(acc_test, accuracy_score(y_test,y_pred_test))
        
        theta_old = np.append(theta_old,np.asarray([theta_new]),axis=0)
        
    return theta_old, acc_train, acc_test

def closed_form_solve(X, y):
    XT = np.transpose(X)
    return np.linalg.inv(XT @ X) @ XT @ y

def closed_form_predict(beta,X_test):
    return np.transpose(beta) @ X_test
    

X_train, X_test, y_train, y_test = 1,1,1,1


