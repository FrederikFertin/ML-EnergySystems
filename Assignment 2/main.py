import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from cluster import RLM, getPriceLevels

# %%
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
n_l = 3

p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_l)

model = RLM(p_levels, p_cuts)
model.calcPmatrix(p_trains, p_levels)

values, iters = model.valueIter(gamma=0.99, maxIter=1000)

profits, socs = model.test(p_test)
del model

plt.plot(profits)
plt.show()

# P = calcPmatrix(df)

# %% Testing

gammas = np.linspace(0.99, 0.9999, 12 - 3)

for ix, n_lev in enumerate(range(3, 12)):
    p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)

    model = RLM(p_levels, p_cuts)
    model.calcPmatrix(p_trains, p_levels)

    values, iters = model.valueIter(gamma=gammas[ix], maxIter=1000)

    profitss, socss = model.test(p_test)
    del model

    plt.title(str("Number of price categories/levels: " + str(n_lev)))
    plt.scatter(np.arange(len(p_test)), p_test, color='black', alpha=0.1, s=0.1)
    plt.plot(profitss)
    plt.ylabel("Total profit")
    plt.xlabel("Hour of trading in test market")
    plt.show()

# plt.plot(socs)
# plt.show()


























