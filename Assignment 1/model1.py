import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from createXY import prepData

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

def feature_selection_linear(data):
    feature_list = data.columns.values
    y = np.array(data['production'])
    feature_list = np.delete(feature_list, np.where(feature_list=='production'))
    X = np.array(data[feature_list])
    # Splitting into train, test and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=False)
    
    # Creating a list of the indeces of features added to the model
    added_features = [np.where(feature_list == 'ones')[0][0]]
    mse_list = []
    while True:
        beta = closed_form_fit(X_train[:,added_features], y_train)
        y_pred = closed_form_predict(beta, X_val[:,added_features])
        mse = mean_squared_error(y_val, y_pred)
        mse_list.append(mse)
        # Looping through every feature and comparing them
        loop_mse_list = []
        for i in range(len(feature_list)):
            # Checking if feature is already added and breaks the loop if true
            if feature_list[i] in feature_list[added_features]:
                loop_mse_list.append(0)
                continue
            testing_features = added_features[:]
            testing_features.append([i][0])
            beta = closed_form_fit(X_train[:,testing_features], y_train)
            y_pred = closed_form_predict(beta, X_val[:,testing_features])
            mse = mean_squared_error(y_val, y_pred)
            loop_mse_list.append(mse-mse_list[-1])
        # Breaking the loop if all added features add nothing/make it worse
        if all(val >= 0 for val in loop_mse_list) or len(feature_list) == len(added_features):
            break

        # Getting the value and index of the highest reduction
        #mse_list.append(min(loop_mse_list))
        best_feature = min(range(len(loop_mse_list)), key=loop_mse_list.__getitem__)
        added_features.append(best_feature)
    
    return feature_list[added_features], mse_list


#%% Data set
data = prepData()

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
y = np.array(data['production'])
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


sel_features, mse_list = feature_selection_linear(data)

# Predicting using the chosen features
y = np.array(data['production'])
X = np.array(data[sel_features])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=False)

beta = closed_form_fit(X_train, y_train)
y_pred = closed_form_predict(beta, X_val)
mse = mean_squared_error(y_val, y_pred)

#print(mse_list)
# Last model is the one additional error metrics are calculated on
MSE = mean_squared_error(y_val, y_pred)
R2 = r2_score(y_val, y_pred)
MAE = mean_absolute_error(y_val, y_pred)

# Plotting
if plotting:
    plt.scatter(data['wind_speed [m/s]'],data['production'],s=1)
    plt.scatter(X_test[:,1],y_pred,s=10)
    plt.xlabel('Wind speed')
    plt.ylabel('Production')
    plt.show


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


#%% Step 7.1
Plotting = False

if plotting:
    n_clusters = 2
    kmeans = KMeans(n_clusters = n_clusters,
                    n_init='auto',
                    random_state = 1)
    features = ['ones', 'wind_speed [m/s]','temperature [C]','pressure [hPa]','past_prod']
    #features = ['ones', 'wind_speed [m/s]']
    cluster_data = data[features]
    
    kmeans.fit(cluster_data)
    center_locations = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    data['label'] = cluster_labels
    
    sets = []
    for i in range(n_clusters):
        sets.append(data[data['label']==i])
        plt.scatter(sets[i]['wind_speed [m/s]'], sets[i]['production'],s=1)
    plt.show()
else:
    kmeans.fit(X_train)
    center_locations = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    test_labels = kmeans.predict(X_test)
    X_test_cluster = X_test
    # Temporarily adding y_labels to data to divide them by label
    X_test_cluster = np.append(X_test_cluster,np.reshape(y_test,(len(y_test),1)),axis=1).shape
    # Adding test labels to X_test
    X_test_cluster = np.append(X_test,np.reshape(test_labels,(len(test_labels),1)),axis=1)
    
    
    X_sets = []
    y_sets = []
    for i in range(n_clusters):
        #sets.append(data[data['label']==i])
        # Picking sets based on label
        mask = (X_test_cluster[:, -1] == i)
        temp_set = X_test_cluster[mask, :]
        # Removing label column again
        temp_set = np.delete(temp_set, -1,1)
        X_sets.append(np.delete(temp_set, -2,1))









































