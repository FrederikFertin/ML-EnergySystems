import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#%%
class ReinforcementModel:    
    
    def __init__(self, prices, socs = np.linspace(0,500,6), actions = np.array([-100,0,100])):
        self.socs = socs
        self.n_socs = len(self.socs)
        self.actions = actions
        self.n_actions = len(self.actions)
        self.setPriceLevels()
        self.P = np.ones(
            (len(self.price_levels),len(self.price_levels))
             ) / len(self.price_levels)
        self.calcRewardMatrix()
    
    def setPriceLevels(self):
        self.price_levels = [-1,0,1]
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
    
    def calcRewardMatrix(self):
        self.R = np.zeros(
            (self.levels, self.n_socs, self.n_actions)
            )
        
        for (l,level) in enumerate(self.R):
            for (s, soc) in enumerate(level):
                for (a, action) in enumerate(soc):
                    if (s == 0 and a == 0)\
                        or (s == self.n_socs-1 and a == self.n_actions-1):
                        self.R[l,s,a] = -np.inf
                    else:
                        self.R[l,s,a] = self.price_levels[l] * self.actions[a]

    def MDP(self, gamma = 0.9, maxIter = 1000):
        self.R = self.R
        last_val = np.zeros(self.R.shape)
        values = np.zeros(self.R.shape)
        iters = 0
        policy = np.ones(self.R.shape)*10
        
        while True:
            for p1 in range(self.levels):
                for s1 in range(self.n_socs):
                    best_val_sum = -np.inf
                    for a in range(self.n_actions):
                        # Calculating sum
                        val_sum = 0
                        # Either make cost of going to impossible soc infinite or
                        # Limit charge/discharge at edge state
                        for p2 in range(self.n_socs):
                            for s2 in range(self.n_socs):
                                val_sum += gamma*self.calcProbability([p1,s1],a,[p2,s2])*max(last_val[p2,s2,[0,1,2]])
                        if val_sum > best_val_sum:
                            policy[p1,s1] = a
                            best_val_sum = val_sum
                    # end for
                    values[p1,s1,a] = self.R[p1,s1,policy[p1,s1]] + val_sum
            if values==last_val or iters >= maxIter:
                break
            last_val = values.copy()
            iters += 1
        
        self.values = values

#%%



"""
def calcStateValues(socs,actions,prices,gamma,P):
    V = np.zeros([len(socs),len(prices),len(actions)])
    for (i,soc) in enumerate(V):
        for (j,price) in enumerate(soc):
            for (k,action) in enumerate(price):
                P = calcProbability([price,soc], action, state2, P)
                V[i,j,k] = gamma*P[i,j,k]*V[i,j,k]
"""

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

soc = np.linspace(0, 500, 6)
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

actions = np.array([-100,0,100])

#P = calcPmatrix(df)






