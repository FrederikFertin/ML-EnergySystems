import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#%%
class RLM:    
    
    def __init__(self, socs = np.linspace(0,500,6), actions = np.array([-100,0,100])):
        self.socs = socs
        self.n_socs = len(self.socs)
        self.actions = actions
        self.n_actions = len(self.actions)
        self.setPriceLevels()
        self.P = np.ones(
            (len(self.price_levels),len(self.price_levels))
             ) / len(self.price_levels)
        self.calcRmatrix()
    
    def setPriceLevels(self):
        self.price_levels = [1,2,3]
        self.levels = len(self.price_levels)
    
    def displayInfo(self):
        print(self.socs)
        print(self.actions)
    
    def calcPmatrix(self,df):
        C = np.zeros((3,3))

        j = 0
        for i in df.Discrete:
            C[j+1,int(i+1)] += 1
            j = int(i)

        self.P = C/len(df.Discrete)*3
    
    def calcProbability(self, s1, a, s2):
        """
        Args: 
            s1: list of state components for "current state" [price, SOC]
            a: charging rate (-100, 0 or 100)
            s2: list of state components for "next state" [price, SOC]
            P: probability matrix for price levels
        
        Returns:
            Probability of going from s1 to s2 given the action.
        """
        
        if s1[1] + a == s2[1]:
            return self.P[s1[0]+1, s2[0]+1]
        else:
            return 0
    
    def calcRmatrix(self):
        
        self.R = np.zeros(
            (self.levels, self.n_socs, self.n_actions)
            )
        
        for (l,level) in enumerate(self.R):
            for (s, soc) in enumerate(level):
                for (a, action) in enumerate(soc):
                    if (s == 0 and a == 0)\
                        or (s == self.n_socs-1 and a == self.n_actions-1):
                        self.R[l,s,a] = - np.inf
                    else:
                        self.R[l,s,a] = - self.price_levels[l] * self.actions[a]

    def valueIter(self, gamma = 0.9, maxIter = 1000):
        last_val = np.zeros((self.levels, self.n_socs))
        values = np.zeros((self.levels, self.n_socs))
        policy = np.zeros((self.levels, self.n_socs))
        iters = 0
        while True:
            for p in range(self.levels):
                for s in range(self.n_socs):
                    vals = []
                    for a in range(self.n_actions):
                        # Calculating sum
                        val_sum = self.R[p,s,a]
                        # Either make cost of going to impossible soc infinite or
                        # Limit charge/discharge at edge state
                        for i in range(len(self.P[p])):
                            if s+a-1 >= 0 and s+a-1 < self.n_socs:
                                val_sum += gamma * self.P[p,i] * last_val[i,s+a-1]
    
                        vals.append(val_sum)
                    values[p,s] = max(vals)
                    policy[p,s] = self.actions[np.argmax(vals)]
                    
            if np.all(values==last_val) or iters >= maxIter:
                break
            last_val = values.copy()
            iters += 1
            #print('test')
        self.V = values
        self.Pi = policy
        return values, iters
    
    def test(self, df_test):
        profits = [0]
        socs = [200]
        for t in range(len(df_test)):
            p = int(df_test['Discrete'].iloc[t])
            s = np.where(self.socs == socs[-1])[0][0]
            
            a = model.Pi[p,s]
            price = df_test['Spot'].iloc[t]
            profits.append(profits[-1] + df_test['Spot'].iloc[t] * (-a))

            socs.append(socs[-1] + a)
        
        return profits, socs


#%%
trainsize = 0.75
cwd = os.getcwd()

filename = os.path.join(cwd,'Prices.csv')

df = pd.read_csv(filename)
df["Hour"] = df.index.values%24
prices = df.Spot
t_of_day = df.Hour


p_train = prices[:int(len(prices)*(trainsize))]
p_test = prices[int(len(prices)*(trainsize)):]

#%%
soc = np.linspace(0, 500, 6)
actions = np.array([-100,0,100])
price_levels = np.quantile(prices,[1/3,2/3])

df["Discrete"] = df["Spot"]
df["Discrete"].loc[df["Spot"] >= price_levels[1]] = 1
df["Discrete"].loc[df["Spot"] < price_levels[1]] = 0
df["Discrete"].loc[df["Spot"] < price_levels[0]] = -1


plt.scatter(df.index,df.Spot,s=0.5,alpha=0.3)
plt.title("Spot prices in 2021 and 2022")
plt.axhline(price_levels[0],color='black',linestyle='--')
plt.axhline(price_levels[1],color='black',linestyle='--')
plt.show()

high = df["Spot"].loc[df["Discrete"] == 1].mean()
medium = df["Spot"].loc[df["Discrete"] == 0].mean()
low = df["Spot"].loc[df["Discrete"] == -1].mean()


#%%
df_train = df.iloc[:180*24,:]
df_test = df.iloc[180*24:200*24,:]

p_train = df_train['Spot']
p_test = df_test['Spot']

high = df["Spot"].loc[df["Discrete"] == 1].mean()
medium = df["Spot"].loc[df["Discrete"] == 0].mean()
low = df["Spot"].loc[df["Discrete"] == -1].mean()

model = RLM()

model.calcPmatrix(df)

values, iters = model.valueIter(gamma = 0.99, maxIter = 1000)



#P = calcPmatrix(df)

#%% Testing 
profits, socs = model.test(df)

plt.plot(profits)
plt.show()

plt.plot(socs)
plt.show()


"""
profits = [0]
soc = 200
for t in range(len(df_test)):
    p = int(df_test['Discrete'].iloc[t])
    s = np.where(model.socs == soc)[0][0]
    
    a = model.Pi[p,s]
    price = df_test['Spot'].iloc[t]
    profits.append(profits[-1] + df_test['Spot'].iloc[t] * (-a))

    soc += a
    
plt.plot(profits)
"""























