# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 15:04:11 2023

@author: Frede
"""

import unit_commitment as uc
import sample_gen as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

multipliers = np.linspace(0.1, 5, 200)

samples = sg.generate_samples(multipliers=multipliers)

X = []
y = []

for sample in samples:
    print(sample)
    X_sample, y_sample = uc.uc(samples[sample])
    X.append(X_sample)
    y.append(y_sample)

