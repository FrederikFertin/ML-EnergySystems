import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from rlm_discrete import RLM_discrete
from rlm_continuous import RLM_continuous
from functions import getPriceLevels, optimalBidding, cf_fit, continual_test, create_train_test
import seaborn as sns
import plot


#%% Preample
cwd = os.getcwd()
filename = os.path.join(cwd, 'Prices.csv')
df = pd.read_csv(filename)
df["Hour"] = df.index.values % 24
prices = df.Spot
t_of_day = df.Hour

# %% Creation of train and test data set
train_days = 180
test_days = 50
p_train, p_test = create_train_test(df, train_days, len_of_train = 180, len_of_test = 50)

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

#%% Fitted value iteration - deterministic
model_cont = RLM_continuous()
iters, thetas = model_cont.fitted_value_iteration(p_train, nSamples = 2000, maxIter=100, gamma = 0.96)
p, s = model_cont.test(p_test, plot = True)

#%% Plot theta 'convergence'
fig, ax1 = plt.subplots()

color = ['tab:red', 'tab:blue']
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Theta SOC weight', color=color[0])
ax1.plot(thetas[0], color=color[0])
ax1.tick_params(axis='y', labelcolor=color[0])

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Theta Price weight', color=color)  # we already handled the x-label with ax1
ax2.plot(thetas[1], color=color, alpha=1)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title(str("Fitted value iteration to obtain " + r'$\theta$' + "-values, " + r'$\gamma$ = 0.96'))
plt.show()

#%% Fitted value iteration - with k-sampling
n_lev = 30
p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)
model_cont.calc_k_samplers(p_train,p_levels, p_trains)
model_cont.fitted_value_iteration_k_sampling(p_train, maxIter=100, gamma=0.96, nSamples = 1000)
p, s = model_cont.test(p_test, plot = True)

#%% Continually updating model
train_days = 180
test_days = 50
length_train = 180 # Good performance
length_test = 50 # Good performance
#length_train = 15
#length_test = 1

model_cont = RLM_continuous()
profits, socs = continual_test(df, model_cont, test_days = test_days, length_train = length_train, length_test = length_test)
model_cont.plot_test_results(profits)

model_cont2 = RLM_continuous(sampling = True)
profits, socs = continual_test(df, model_cont2, gamma=0.96, test_days = test_days, length_train = length_train, length_test = length_test)
model_cont2.plot_test_results(profits)

model_disc = RLM_discrete()
profits, socs = continual_test(df, model_disc, test_days = test_days, length_train = length_train, length_test = length_test, n_levels = 24)
model_cont.plot_test_results(profits)


# %% Continuous state space - additional method
""" We acknowledge that the prices are a real life continuous variable.
The state of charge is deterministic given the action, 
so this is kept discrete and non-stochastic. 
The action has no impact on the price, as we are a price-taker in the market.
Therefore we can model the price as a regression based on previous price states. """
n_l = 30
p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_l)

# Distributions for each price level used in k-sampling
for aLevel in p_levels:
    p_now = p_train.loc[p_trains==aLevel]
    p_chooser = (p_trains==aLevel).shift(1)
    p_chooser[0] = (p_trains==aLevel).iloc[-1]
    p_next = p_train.loc[p_chooser]
    p_diff = pd.Series(p_next.values - p_now.values)
    sns.displot(p_diff, kind='kde', bw_adjust=.5)
plt.show()

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

