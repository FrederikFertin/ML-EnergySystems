import pandas as pd
import os, csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv

#%% Feature data and actual wind power production:
cwd = os.getcwd()

""" Load actual wind power from cwd """
temp_dir = os.path.join(cwd, './Actual wind power.csv')
df = pd.read_csv(temp_dir,sep=";")

### Create time index to match feature data
df["HourUTC"] = df['Date'].astype(str) +" "+ df["Time"]
df['HourUTC'] = df['HourUTC'].apply(lambda x: str(x.replace('.','-')))
df['HourUTC'] = pd.to_datetime(df['HourUTC'], format='%d-%m-%Y %H:%M')
df = df[['HourUTC','Actual']]
df['HourUTC'] = df['HourUTC'].dt.tz_localize(None)
df = df.drop_duplicates('HourUTC',keep='first')
df.set_index('HourUTC', inplace=True)

### Sort out DST issues with duplicate values and missing observation (2022-01-01 00:00)
df1Jan = np.mean([df.loc['2022-01-01 01:00']['Actual'],df.loc['2022-12-31 23:00']['Actual']])
line = pd.DataFrame({"Actual": df1Jan}, index=[datetime.strptime('2022-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')])
df =pd.concat([df, line], ignore_index=False)
df = df.sort_index()
df = df.iloc[0:len(df['Actual'])-1,:]
""" Power production is already normalized """


""" Load data files from folder """
temp_dir = os.path.join(cwd, './data.csv')
data = pd.read_csv(temp_dir)
data.set_index('HourUTC', inplace=True)

attributeNames = np.asarray(data.columns)
dfs = data.copy()


for i in range(0,len(attributeNames)-1):
    mu_dfs = np.mean(dfs[attributeNames[i]])
    std_dfs = np.std(dfs[attributeNames[i]])
    dfs[attributeNames[i]] = (dfs[attributeNames[i]]-mu_dfs)/std_dfs


# Formatting
y = np.array(df['Actual'].values)
x1 = np.ones(len(y))
cols = dfs.columns
X = np.matrix([x1,dfs[cols[0]],dfs[cols[1]],dfs[cols[3]]]).T


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)




# Closed form linear regression:
beta = np.array(inv(X_train.T @ X_train) @ X_train.T @ y_train).reshape(-1)
y_pred = np.array(X_test @ beta).reshape(-1)
mse = mean_squared_error(y_test, y_pred)


## Creating array of day ahead prices in DK2 area
Day_ahead_data = pd.read_excel("./Day-ahead_price.xlsx")
Day_ahead_all = np.array(Day_ahead_data.loc[:,"SpotPriceEUR"]) 
ahead_DK2 = np.array(Day_ahead_all[8:len(Day_ahead_all):7])
# Deleting data from after 2022
ahead_DK2 = ahead_DK2[:17518]

## Extracting Balancing prices 
Up21=[]
with open("./Up-regulation_price_2021.csv", 'r') as file:
    file.readline()
    for line in file.readlines():
        elements = line.split(",")
        price = float(elements[4].translate({ord(i): None for i in '\n""'}))
        Up21.append(price)
        
Up21 = np.array(Up21[2:])

Up22=[]
with open("./Up-regulation_price_2022.csv", 'r') as file:
    file.readline()
    for line in file.readlines():
        elements = line.split(",")
        price = float(elements[4].translate({ord(i): None for i in '\n""'}))
        Up22.append(price)
        
Up22 = np.array(Up22)

Up = np.concatenate((Up21, Up22))

Down21=[]
with open("./Down-regulation_price_2021.csv", 'r') as file:
    file.readline()
    for line in file.readlines():
        elements = line.split(",")
        price = float(elements[4].translate({ord(i): None for i in '\n""'}))
        Down21.append(price)
        
Down21 = np.array(Down21[2:])

Down22=[]
with open("./Down-regulation_price_2022.csv", 'r') as file:
    file.readline()
    for line in file.readlines():
        elements = line.split(",")
        price = float(elements[4].translate({ord(i): None for i in '\n""'}))
        Down22.append(price)
        
Down22 = np.array(Down22)

Down = np.concatenate((Down21, Down22))

## Revenue Calculation
def revenue_calc(y_test,y_pred):
   act_pow = y_test[:-2]
   pred_pow = y_pred[:-2]
   start= len(y_train)
   stop = start+len(pred_pow)-1
   ahead_rev = ahead_DK2[start:stop]
   Down_rev = Down[start:stop]
   Up_rev = Up[start:stop]
   revenue = 0
   for i in range(0,len(pred_pow)-1):
      revenue += ahead_rev[i]* y_pred[i]
      if (act_pow[i]-pred_pow[i])>0:
         revenue += Down_rev[i]*(act_pow[i]-pred_pow[i])
      elif (act_pow[i]-pred_pow[i])<0:
         revenue += Up_rev[i]*(act_pow[i]-pred_pow[i])
   revenue = round(revenue / len(pred_pow) *24, 2)
   return("Average daily revenue in EUR: " + str(revenue))


revenue_calc(y_test, y_pred)



# Model 2

import gurobipy as gp
from gurobipy import GRB

act_pow = y_train
BalUp = Up[:len(act_pow)]
BalDown = Down[:len(act_pow)]
ahead_price = ahead_DK2[:len(act_pow)]
      
   
model = gp.Model()

# Definition of decision variable
opt_pred = model.addVars(len(act_pow),vtype=GRB.INTEGER, name="opt_pred")

# Definition of binary variable
is_neg = model.addVars(len(act_pow), vtype=GRB.BINARY, name="is_neg")

# Definition of constraints (If maximum power production is known, it can be added)
model.addConstrs(
   (opt_pred[i]>=0) for i in range(len(act_pow)))

model.addConstrs(
   (opt_pred[i]<=1) for i in range(len(act_pow)))

# Constants

M = 1  # smallest possible given bounds on x and y

# Model if opt_pred[i] > act_pow[i], then is_positive = 1, otherwise is_positive = 0

model.addConstrs( opt_pred[i] >=  act_pow[i] - M * (1-is_neg[i]) for i in range(len(act_pow)))
model.addConstrs(opt_pred[i] <= (act_pow[i] + M * is_neg[i]) for i in range(len(act_pow)-1))

# Definition of objective function
model.setObjective((
   gp.quicksum(ahead_price[i]* opt_pred[i] + 
               (1-is_neg[i]) * BalDown[i] * (act_pow[i]-opt_pred[i]) +
               is_neg[i] * BalUp[i] * (act_pow[i]-opt_pred[i]) 
               for i in range(len(act_pow)))
   ),GRB.MAXIMIZE)

model.optimize()

opt_list = []
for v in model.getVars():
   if "opt_pred" in v.VarName:
      opt_list.append(v.x)

ytrain_new = np.array(opt_list)
   

# Closed form linear regression:
beta = np.array(inv(X_train.T @ X_train) @ X_train.T @ ytrain_new).reshape(-1)
y_bit = np.array(X_test @ beta).reshape(-1)
mse = mean_squared_error(y_test, y_pred)
   
print("MSE: ",mse)

revenue_calc(y_test,y_bit)

# Other regression forms:


