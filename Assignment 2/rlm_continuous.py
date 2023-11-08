import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt

class RLM_continuous:
        
    def __init__(self, plevels = [0,1,2], p_cuts = [0.5,1.5], socs = np.linspace(0,500,6), actions = np.array([-100,0,100])):
        self.socs = socs
        self.minsoc = min(self.socs)
        self.maxsoc = max(self.socs)
        self.n_socs = len(self.socs)
        
        self.actions = actions
        self.n_actions = len(self.actions)
        
        self.price_cuts = p_cuts
        self.price_levels = plevels
        self.n_levels = len(self.price_levels)
        
        self.theta = None

    def phi(self,state):
        # s.append(s[1]**2)
        # s.append(np.log(s[0]))
        return state
    
    def cf_fit(self, X, y):
        """ Closed form predict """
        XT = np.transpose(X)
        return np.linalg.inv(XT @ X) @ XT @ y
    
    def createX(self,y):
        ones = np.ones(len(y))
        hour_1 = np.insert(y[:-1],0,y[0])
        hour_2 = np.insert(hour_1[:-1],0,hour_1[0])
        
        X = np.array([ones, hour_1, hour_2]).T
        
        return X

    def fitted_value_iteration(self, prices_train, nSamples = 500, gamma = 0.999, maxIter = 1000):
        y = prices_train.values
        X = self.createX(y)
        
        beta = self.cf_fit(X,y)
        
        n = nSamples
        t_samples = [random.randint(0,len(y)-1) for i in range(n)]
        samples = [[random.randint(0,5)*100, y[t_samples[i]]] for i in range(n)]
        theta = np.zeros(len(samples[0]))
        yy = np.zeros(n)
        iters = 0
        
        while True:
            iters += 1
            if iters > maxIter: break
            print("Iteration: ", iters)
            for i in range(n):
                q = np.zeros(len(self.actions))
                s = samples[i][0]
                p = samples[i][1]
                for ix, a in enumerate(self.actions):
                    s_prime = [s + a, X[t_samples[i]] @ beta]
                    if s_prime[0] > 500 or s_prime[0] < 0:
                        R = -np.inf
                    else:
                        R = - a * p
                    q[ix] = R + gamma * (self.phi(s_prime) @ theta)
                yy[i] = max(q)
            theta = self.cf_fit(self.phi(samples),yy)
        self.theta = theta
        
        return iters
    
    def plot_test_results(self):
        plt.title(str("\'Continuous\' pricing: "))
        plt.plot(np.array(self.test_profits))
        #plt.plot(self.socs)
        plt.ylabel("Total profit")
        plt.xlabel("Hour of trading in test market")
        plt.show()
    
    def test(self, p_test, plot = False):
        if self.theta.any() == None:
            print("Need to run fitted_value_iteration first to define theta")
        else:
            profits = [0]
            socs = [200]
            
            for t in range(len(p_test)):
                p = p_test.iloc[t]
                
                s = socs[-1]
                
                V = []
                for a in self.actions:
                    if s + a > 500 or s + a < 0:
                        V.append(-np.inf)
                    else:
                        state = np.array([s+a,p])
                        V.append(-p*a + self.phi(state) @ self.theta)
                a = self.actions[np.argmax(V)]
                
                profits.append(profits[-1] + p * (-a))
                socs.append(socs[-1] + a)
            
            self.test_profits = profits
            self.test_socs = socs
            
            if plot:
                self.plot_test_results()
            
            return profits, socs
    
