import pandas as pd
import numpy as np
import random

def generate_samples(multipliers = [0.5,1,2,5]):
    demand = pd.read_csv('system_data/demand.csv')
    samples = pd.read_csv('system_data/samples.csv', header=None)
    demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])
    P_load_max = np.max(demand,axis=0)
    samples = samples*P_load_max
    samples_dict = {}
    for i in multipliers:
        samples_dict[i] = samples.copy()*i
    return samples_dict

def pick_samples(df, randseed = random.random()):
    random.seed(randseed)
    sample = np.empty([24,91])
    for i in range(24):
        sample[i,:] = df.iloc[random.randint(0,len(df))]
    return sample

