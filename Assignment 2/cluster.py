import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split

#%%
class RLM:    
    
    def __init__(self, plevels = [0,1,2], p_cuts = [0.5,1.5], socs = np.linspace(0,500,6), actions = np.array([-100,0,100])):
        self.socs = socs
        self.n_socs = len(self.socs)
        self.actions = actions
        self.n_actions = len(self.actions)
        self.price_cuts = p_cuts
        self.setPriceLevels(plevels)
        self.P = np.ones(
            (self.n_levels,self.n_levels)
             ) / self.n_levels
        self.calcRmatrix()
    
    def setPriceLevels(self, plevels):
        self.price_levels = plevels
        self.n_levels = len(self.price_levels)
    
    def displayInfo(self):
        print(self.socs)
        print(self.actions)
    
    def calcPmatrix(self,prices,price_levels):
        prices = np.round(prices,2)
        price_levels = np.round(price_levels,2)
        
        C = np.zeros((self.n_levels,self.n_levels))
        price_level_count = np.zeros(self.n_levels)
        last_price = prices.iloc[-1]
        for (ix, current_price) in enumerate(prices):
            if ix == 0:
                current_price = prices.iloc[0]
            from_price = np.where(np.array(price_levels) == last_price)[0][0]
            to_price = np.where(np.array(price_levels) == current_price)[0][0]
            C[from_price,to_price] += 1
            last_price = current_price
            price_level_count[from_price] +=1
        
        #self.P = C/len(prices)*len(price_levels)
        self.P = C/price_level_count
    
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
            (self.n_levels, self.n_socs, self.n_actions)
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
        last_val = np.zeros((self.n_levels, self.n_socs))
        values = np.zeros((self.n_levels, self.n_socs))
        policy = np.zeros((self.n_levels, self.n_socs))
        iters = 0
        while True:
            for p in range(self.n_levels):
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
    
    def test(self, p_test):
        prices = p_test.copy()
        profits = [0]
        socs = [200]
        
        d_prices = p_test.copy()
        for i in range(self.n_levels):
            if i < self.n_levels-1:
                d_prices.loc[(prices >= self.price_cuts[i]) & (prices < self.price_cuts[i+1])] = self.price_levels[i]
            else:
                d_prices.loc[prices >= self.price_cuts[i]] = self.price_levels[i]
        
        for t in range(len(prices)):
            p = np.where(np.round(self.price_levels,2) == np.round(d_prices.iloc[t],2))[0][0]
            
            s = np.where(self.socs == socs[-1])[0][0]
            
            a = model.Pi[p,s]
            price = prices.iloc[t]
            profits.append(profits[-1] + price * (-a))
            socs.append(socs[-1] + a)
        
        return profits, socs

def getPriceLevels(p_train,n_levels):
    prices = p_train.copy()
    p = prices.copy()
    price_cuts = np.quantile(prices, np.linspace(0,1,n_levels+1))[:n_levels]
    
    price_levels = np.zeros(n_levels)
    for i in range(n_levels):
        if i < n_levels-1:
            price_levels[i] = prices.loc[(p >= price_cuts[i]) & (p < price_cuts[i+1])].mean()
            prices.loc[(p >= price_cuts[i]) & (p < price_cuts[i+1])] = price_levels[i]
        else:
            price_levels[i] = prices.loc[p >= price_cuts[i]].mean()
            prices.loc[p >= price_cuts[i]] = price_levels[i]
    
    return prices, price_levels, price_cuts

    

#%%
trainsize = 0.75
cwd = os.getcwd()

filename = os.path.join(cwd,'Prices.csv')

df = pd.read_csv(filename)
df["Hour"] = df.index.values%24
prices = df.Spot
t_of_day = df.Hour


#p_train = prices[:int(len(prices)*(trainsize))]
#p_test = prices[int(len(prices)*(trainsize)):]

#%%
"""
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

"""

#%% Creation of train and test data set
train_days = 180
test_days = 20

df_train = df.iloc[:train_days*24,:]
df_test = df.iloc[train_days*24:(train_days+test_days)*24,:]

p_train = df_train['Spot']
p_test = df_test['Spot']

#%% Singular tests
n_l = 3

p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_l)

model = RLM(p_levels, p_cuts)
model.calcPmatrix(p_trains, p_levels)

values, iters = model.valueIter(gamma=0.99,maxIter=1000)

profits, socs = model.test(p_test)

plt.plot(profits)
plt.show()

#P = calcPmatrix(df)

#%% Testing 
del model
gammas = np.linspace(0.99,0.9999,12-3)

for ix, n_lev in enumerate(range(3,12)):
    p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)
    
    model = RLM(p_levels, p_cuts)
    model.calcPmatrix(p_trains, p_levels)

    values, iters = model.valueIter(gamma=gammas[ix],maxIter=1000)

    profitss, socss = model.test(p_test)
    plt.title(str("Number of price categories/levels: " + str(n_lev)))
    plt.scatter(np.arange(len(p_test)),p_test,color='black',alpha=0.1,s=0.1)
    plt.plot(profitss)
    plt.ylabel("Total profit")
    plt.xlabel("Hour of trading in test market")
    plt.show()
    del model

#plt.plot(socs)
#plt.show()
























