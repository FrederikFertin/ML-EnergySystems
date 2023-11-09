import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from rlm_discrete import RLM_discrete
from rlm_continuous import RLM_continuous
from cluster import getPriceLevels, optimalBidding, cf_fit, cf_predict
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LogisticRegression
import random
import seaborn as sns
from distfit import distfit
import plot


#%% Preample
trainsize = 0.75
cwd = os.getcwd()

filename = os.path.join(cwd, 'Prices.csv')

df = pd.read_csv(filename)
df["Hour"] = df.index.values % 24
prices = df.Spot
t_of_day = df.Hour

# p_train = prices[:int(len(prices)*(trainsize))]
# p_test = prices[int(len(prices)*(trainsize)):]

# %%
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

# %% Creation of train and test data set
train_days = 180
test_days = 50

df_train = df.iloc[:train_days * 24, :]
df_test = df.iloc[train_days * 24:(train_days + test_days) * 24, :]

p_train = df_train['Spot']
p_test = df_test['Spot']

# %% Singular tests
n_l = 30

p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_l)

model = RLM_discrete(p_levels, p_cuts)
model.calcPmatrix(p_trains)

values, iters = model.valueIter(gamma=0.9999, maxIter=1000)

profits, socs = model.test(p_test)
#del model

plt.plot(profits)
plt.show()

# P = calcPmatrix(df)

# %% Testing

gammas = np.linspace(0.99, 1, 12 - 3)

for ix, n_lev in enumerate(range(3, 12)):
    p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)

    model = RLM_discrete(p_levels, p_cuts)
    model.calcPmatrix(p_trains)

    values, iters = model.valueIter(gamma=gammas[ix], maxIter=1000)

    profitss, socss = model.test(p_test)
    del model

    plt.title(str("Number of price categories/levels: " + str(n_lev)))
    plt.scatter(np.arange(len(p_test)), p_test, color='black', alpha=0.1, s=0.1)
    plt.plot(profitss)
    plt.ylabel("Total profit")
    plt.xlabel("Hour of trading in test market")
    plt.show()

#%% Optimal bidding
opt_profits, opt_socs, opt_p_ch = optimalBidding(p_test)

opt_profits_cumulated = [0]
for t in range(1,len(p_test)):
    opt_profits_cumulated.append(opt_profits_cumulated[t-1] + opt_p_ch[t]*p_test.iloc[t])

plt.plot(opt_profits_cumulated)

#%% Plot comparing cumulated profits
plot.plotCompareProfits(
    opt_profits_cumulated,
    profits2=profits,
    labels=["Optimal","Discreet RLM"],
    title="Comparison of cumulated profits for optimal and discrete RLM",
    p_test=p_test,
    )

#%% Fitted value iteration object

model_cont = RLM_continuous()
model_cont.fitted_value_iteration(p_train, nSamples = 2000, maxIter=100, gamma = 0.96)
model_cont.test(p_test, plot = True)


#%% To be used in k-sampling
for aLevel in p_levels:
    p_now = p_train.loc[p_trains==aLevel]
    p_chooser = (p_trains==aLevel).shift(1)
    p_chooser[0] = (p_trains==aLevel).iloc[-1]
    p_next = p_train.loc[p_chooser]
    p_diff = pd.Series(p_next.values - p_now.values)
    print(np.mean(p_diff))
    sns.displot(p_diff, kind='kde', bw_adjust=.5)
plt.show()



# %% Continuous state space - additional method
""" We acknowledge that the prices are a real life continuous variable.
The state of charge is deterministic given the action, 
so this is kept discrete and non-stochastic. 
The action has no impact on the price, as we are a price-taker in the market.
Therefore we can model the price as a regression based on previous price states. """
n_l = 30
p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_l)
model = RLM_discrete(p_levels, p_cuts)
model.calcPmatrix(p_trains)
values, iters = model.valueIter(gamma=1, maxIter=1000)

V = model.V.T
thetas = [cf_fit(np.array([np.ones(len(V[i])), p_levels]).T, V[i]) for i in range(6)]

actions = [-100, 0, 100]
socs = [200]
profits = [0]
for t in range(len(p_test)):
    p = p_test.iloc[t]

    s = socs[-1]

    V = []
    for a in actions:
        if s + a > 500 or s + a < 0:
            V.append(-np.inf)
        else:
            V.append(sum(thetas[int((s + a) / 100)] * [1, p]) - p * a)
    a = actions[np.argmax(V)]

    profits.append(profits[-1] + p * (-a))
    socs.append(socs[-1] + a)

plt.title(str("\'Continuous\' pricing: "))
plt.scatter(np.arange(len(p_test)), p_test, color='black', alpha=0.1, s=0.1)
plt.plot(np.array(profits) / 1000)
plt.plot(socs)
plt.ylabel("Total profit")
plt.xlabel("Hour of trading in test market")
plt.show()




















