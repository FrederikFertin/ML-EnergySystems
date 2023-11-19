from collections import defaultdict
import numpy as np
import gurobipy as grb
import pandas as pd

def uc(demand):
    #%% Load bus system
    pmax = pd.read_csv('system_data/pgmax.csv')
    pmin = pd.read_csv('system_data/pgmin.csv')
    ru = pd.read_csv('system_data/ramp.csv')
    UT = pd.read_csv('system_data/lu.csv')
    DT = pd.read_csv('system_data/ld.csv')
    #demand = pd.read_csv('system_data/demand.csv')
    c_op = pd.read_csv('system_data/cost_op.csv')
    c_st = pd.read_csv('system_data/cost_st.csv')
    PTDF = pd.read_csv('system_data/PTDF.csv')
    busgen = pd.read_csv('system_data/busgen.csv')
    busload = pd.read_csv('system_data/busload.csv')
    fmax = pd.read_csv('system_data/fmax.csv')
    
    
    PTDF = pd.DataFrame(PTDF)
    busgen = pd.DataFrame(busgen)
    busload = pd.DataFrame(busload)
    #demand = pd.DataFrame(demand)
    
    # Convert the PTDF and demand string elements to a matrix
    PTDF_matrix = np.vstack([list(map(float, row[0].split(';'))) for row in PTDF.values])
    busgen_matrix = np.vstack([list(map(float, row[0].split(';'))) for row in busgen.values])
    busload_matrix = np.vstack([list(map(float, row[0].split(';'))) for row in busload.values])
    #demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])
    
    Hg = np.dot(PTDF_matrix, busgen_matrix)
    Hl = np.dot(PTDF_matrix, busload_matrix)
    
    #%% Initialize model
    model = grb.Model()
    
    #%% Model variables
    T = np.shape(demand)[0]
    n_t = T
    
    n_g = np.shape(Hg)[1]
    n_load = np.shape(Hl)[1]
    n_line = np.shape(Hl)[0]
    
    # b denotes on/off status and u denotes start start-up status
    b = model.addVars(n_g, n_t, vtype=grb.GRB.BINARY)
    u = model.addVars(n_g, n_t, vtype=grb.GRB.BINARY)
    
    # how much to produce
    p = model.addVars(n_g, n_t, vtype=grb.GRB.CONTINUOUS)
    
    # slack variables
    #ls = model.addVars(n_t, 1, vtype=grb.GRB.CONTINUOUS)
    #ws = model.addVars(n_t, 1, vtype=grb.GRB.CONTINUOUS)
    model.update()
    
    #%% Generator constraints 
    for t in range(n_t):
        for j in range(n_g):
            # Constraints limiting the production at each generator based on min/max and whether the gen is on.
            model.addConstr(p[j,t] <= pmax.iloc[j].item() *b[j,t])
            model.addConstr(p[j,t] >= pmin.iloc[j].item() *b[j,t])
            if t >= 1:
                # Ramp-up limit:
                model.addConstr(p[j,t]- p[j,t-1] <= ru.iloc[j].item())
                # Ramp-down limit:
                model.addConstr(p[j,t-1]- p[j,t] <= ru.iloc[j].item())
                # u[j,t] == 1 if the generator started in hour t.
                model.addConstr(u[j,t] >= b[j,t]-b[j,t-1])
                
                # The minimal number of hours each generator should run if started.
                min_on_time = min(t + UT.iloc[j].item() - 1, n_t)
                for tau in range(t, min_on_time):
                    model.addConstr(b[j, tau] >= b[j, t] - b[j, t - 1])
                
                # The minimal number of hours each generator should be off if stopped.
                min_off_time = min(t + DT.iloc[j].item() - 1, n_t)
                for tau in range(t, min_off_time):
                    model.addConstr(1 - b[j, tau] >= b[j, t - 1] - b[j, t])
    
    #%% Supply & Demand constraint
    for t in range(n_t):
        # The total production during each hour needs to match the total demand.
        model.addConstr(sum(p[j,t] for j in range(n_g)) == sum(demand[t, i] for i in range(n_load)))
    model.update()
    
    #%% Line loading constraints
    for t in range(n_t):
        for i in range(n_line):
            # The expression for the total load on each line in the network.
            # Based on the positive flow added from the generators and the negative flow added from the loads.
            expr = sum(Hg[i, j] * p[j, t] for j in range(n_g)) - sum(Hl[i, j] * demand[t,j] for j in range(n_load))
            model.addConstr(expr <= fmax.iloc[i].item())
    
    #%% Model Objective
    # The objective is to minimize the cost of power and start-up costs.
    expr = sum(c_op.iloc[j].item() * p[j, t] for j in range(n_g) for t in range(n_t)) +\
        sum(c_st.iloc[j].item() * u[j, t] for j in range(n_g) for t in range(n_t))
    model.setObjective(sense=grb.GRB.MINIMIZE, expr=expr)
    model.update()
    
    #%% Model solving
    opt = model.optimize()
    
    #%% Results loading - needs work
    #creating training data for output y (binary on/off status of generators)
    y = np.zeros((n_g,n_t))       
    #variables = model.getVars()
    for g in range(n_g):
        for t in range(n_t):
            y[g,t] = b[g, t].x
    
    X = demand
    
    return X, y

