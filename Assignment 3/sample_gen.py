import pandas as pd
import numpy as np
import random


def generate_samples(n_samples):
    demand = pd.read_csv('system_data/demand.csv')
    samples = pd.read_csv('system_data/samples.csv', header=None)
    pmax = np.asarray(pd.read_csv('system_data/pgmax.csv'))
    pmin = np.asarray(pd.read_csv('system_data/pgmin.csv'))
    max_gen = np.sum(pmax)
    min_gen = np.min(pmin)
    
    demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])
    
    p_load_max = np.max(demand, axis=0)
    scaled_samples = samples*p_load_max
    
    max_demand = scaled_samples.sum(axis=1).max()
    min_demand = scaled_samples.sum(axis=1).min()
    
    max_scaling = max_gen/max_demand*0.9
    min_scaling = min_gen/min_demand*1.1
    
    
    multipliers = np.linspace(min_scaling, max_scaling, n_samples)
    samples_dict = {}
    for i in multipliers:
        samples_dict[i] = pick_samples(scaled_samples.copy()*i)
    return samples_dict


def pick_samples(df, seed=random.random()):
    random.seed(seed)
    sample = np.empty([24, 91])
    for t in range(24):
        sample[t, :] = df.iloc[random.randint(0, len(df))]
    return sample

