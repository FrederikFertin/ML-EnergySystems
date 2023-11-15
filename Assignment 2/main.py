import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from rlm_discrete import RLM_discrete
from rlm_continuous import RLM_continuous
from functions import getPriceLevels, optimalBidding, cf_fit, continual_test, create_train_test, create_second_train_test
import seaborn as sns
import plot

#%% Preample
cwd = os.getcwd()
filename = os.path.join(cwd, 'Prices.csv')
df = pd.read_csv(filename)
df["Hour"] = df.index.values % 24
prices = df.Spot
t_of_day = df.Hour

#%% Plot entire price data set
_, _, p_cuts = getPriceLevels(prices, 3)
plot.plotPriceData(prices, p_cuts, train_length=180, test_length=50)

# %% Creation of train and test data sets
train_days = 180
test_days = 50
p_train, p_test = create_train_test(df, train_days, len_of_train = 180, len_of_test = 50)

p_train2, p_test2 = create_second_train_test(prices, train_days, test_days)

#%% Finding the best discrete model

#For plotting and comparing many levels
gammas = np.linspace(0.99, 1, 12 - 3)

for ix, n_lev in enumerate(range(3, 12)):
    p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)

    model = RLM_discrete(p_levels, p_cuts)
    model.calcPmatrix(p_trains)

    values, iters = model.valueIter(gamma=gammas[ix], maxIter=1000)

    profitss, socss = model.test(p_test)
    del model

    plt.title(str("Number of price categories/levels: " + str(n_lev)))
    #plt.scatter(np.arange(len(p_test)), p_test, color='black', alpha=0.1, s=0.1)
    plt.plot(profitss)
    plt.ylabel("Total profit")
    plt.xlabel("Hour of trading in test market")
    plt.show()

profits_list = []
for n_lev in [3,7,11]:
    p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)

    model = RLM_discrete(p_levels, p_cuts)
    model.calcPmatrix(p_trains)

    values, iters = model.valueIter(gamma=0.999, maxIter=1000)

    profitss, socss = model.test(p_test)
    profits_list.append(profitss)
    del model

plot.plotCompareProfits(
    profits_list,
    labels=["Price levels: 3","Price levels: 7", "Price levels: 11"],
    title="Gamma = 0.99",
    p_test=p_test
)

# Choose the model with 11 price levels for future reference
profits_discrete = profits_list[2]

#%% Fitted value iteration - deterministic
model_cont = RLM_continuous()
iters, thetas = model_cont.fitted_value_iteration(p_train, nSamples = 2000, maxIter=200, gamma = 0.96)
profits_cont_deter, soc_cont_deter = model_cont.test(p_test, plot = True)

#%% Plot theta 'convergence'
plot.plotThetaConvergence(thetas)

#%% Fitted value iteration - with k-sampling
n_lev = 20
p_trains, p_levels, p_cuts = getPriceLevels(p_train, n_lev)
model_cont.calc_k_samplers(p_train,p_levels, p_trains)
model_cont.fitted_value_iteration_k_sampling(p_train, maxIter=100, gamma=0.96, nSamples = 1000)
profits_cont_sampling, s_cont_sampling = model_cont.test(p_test, plot = True)

#%% Continually updating model (online)
length_train = 1 # Good performance
length_test = 1 # Good performance

model_cont = RLM_continuous()
profits_cont_deter_online, socs = continual_test(df, model_cont, test_days = test_days, length_train = length_train, length_test = length_test)
model_cont.plot_test_results(profits_cont_deter_online)

model_cont2 = RLM_continuous(sampling = True)
profits_cont_k_online, socs = continual_test(df, model_cont2, gamma=0.96, test_days = test_days, length_train = length_train, length_test = length_test)
model_cont2.plot_test_results(profits_cont_k_online)

model_disc = RLM_discrete()
profits_disc_online, socs = continual_test(df, model_disc, test_days = test_days, length_train = length_train, length_test = length_test, n_levels = 6)
model_cont.plot_test_results(profits_disc_online)

model_disc2 = RLM_discrete()
profits, socs = continual_test(df, model_disc, test_days = test_days, length_train = train_days, length_test = test_days, n_levels = 24)
model_cont.plot_test_results(profits)

p_train, p_test = create_train_test(df, train_days, len_of_train = train_days, len_of_test = test_days)
opt_profits, opt_socs, opt_p_ch = optimalBidding(p_test)
opt_profits_cumulated = [0]
for t in range(1,len(p_test)):
    opt_profits_cumulated.append(opt_profits_cumulated[t-1] + opt_p_ch[t]*p_test.iloc[t])

plot.plotCompareProfits(
    opt_profits_cumulated,
    profits2=profits,
    labels=["Optimal","Discreet RLM"],
    title="Comparison of cumulated profits for optimal and discrete RLM",
    p_test=p_test
    )

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

#%% Optimal bidding
opt_profits, opt_socs, opt_p_ch = optimalBidding(p_test)

opt_profits_cumulated = [0]
for t in range(1,len(p_test)):
    opt_profits_cumulated.append(opt_profits_cumulated[t-1] + opt_p_ch[t]*p_test.iloc[t])

#%% Plot comparing cumulated profits 2021
plot.plotCompareProfits([
    opt_profits_cumulated,
    profits_discrete,
    profits_cont_deter,
    profits_disc_online[:-50],
    ],
    labels=["Optimal","Discrete", "Continuous","Disc. online", "Cont. online"],
    title="Comparison of cumulated profits 2021 (50 days)",
    p_test=p_test,
)

#%%################## 2022 ########################
#%% Discrete model
n_lev2 = 8
p_trains2, p_levels2, p_cuts2 = getPriceLevels(p_train2, n_lev2)

model = RLM_discrete(p_levels2, p_cuts2)
model.calcPmatrix(p_trains2)

values2, iters2 = model.valueIter(gamma=0.999, maxIter=1000)

profits_disc_2022, socs_disc_2022 = model.test(p_test2)

#%% Continuous model (deterministic)
model_cont_2022 = RLM_continuous()
iters, thetas = model_cont.fitted_value_iteration(p_train2, nSamples = 2000, maxIter=200, gamma = 0.94)
profits_cont_deter_2022, soc_cont_deter_2022 = model_cont.test(p_test2, plot=True)

#%% Online discrete model
length_train = 1 # Good performance
length_test = 1 # Good performance

model_disc_2022 = RLM_discrete()
profits_disc_online_2022, socs = continual_test(df.iloc[8760:], model_disc_2022, test_days = test_days, length_train = length_train, length_test = length_test, n_levels = 7)
model_cont.plot_test_results(profits_disc_online_2022)

#%% Optimal bidding
opt_profits_2022, opt_socs_2022, opt_p_ch_2022 = optimalBidding(p_test2)

opt_profits_cumulated_2022 = [0]
for t in range(1,len(p_test2)):
    opt_profits_cumulated_2022.append(opt_profits_cumulated_2022[t-1] + opt_p_ch_2022[t]*p_test2.iloc[t])

#%% Plot comparing cumulated profits 2021
plot.plotCompareProfits([
    opt_profits_cumulated_2022,
    profits_disc_2022,
    profits_cont_deter_2022,
    profits_disc_online_2022[:-50],
    # profits_cont_deter_online_2022,
    ],
    labels=["Optimal","Discrete", "Continuous","Disc. online", "Cont. online"],
    title="Comparison of cumulated profits 2022 (50 days)",
    p_test=p_test2,
)



































