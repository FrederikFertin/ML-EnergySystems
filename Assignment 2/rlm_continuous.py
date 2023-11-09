import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


class RLM_continuous:
        
    def __init__(self, socs = np.linspace(0,500,6), actions = np.array([-100,0,100]), sampling = False):
        
        self.model_type = "continuous"
        self.sampling = sampling
        
        self.socs = socs
        self.minsoc = min(self.socs)
        self.maxsoc = max(self.socs)
        self.n_socs = len(self.socs)
        
        self.actions = actions
        self.n_actions = len(self.actions)
        
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

    def fitted_value_iteration(self, prices_train, nSamples = 1000, gamma = 0.96, maxIter = 200):
        y = prices_train.values
        X = self.createX(y)
        
        beta = self.cf_fit(X,y)
        
        n = nSamples
        t_samples = [random.randint(0,len(y)-1) for i in range(n)]
        samples = [[random.randint(0,5)*100, y[t_samples[i]]] for i in range(n)]
        theta = np.zeros(len(samples[0]))
        prev_theta = np.zeros(len(samples[0]))
        yy = np.zeros(n)
        iters = 0
        
        thetas = []
        
        while True:
            iters += 1
            if iters > maxIter: break
            #print("Iteration: ", iters)
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
            if np.array_equal(theta, prev_theta):
                break
            prev_theta = theta
            thetas.append(prev_theta)
        self.theta = theta
        
        return iters, np.array(thetas).T
    
    def calc_k_samplers(self, prices, p_levels, disc_prices):
        self.k_samplers = []
        self.price_levels = p_levels
        for aLevel in p_levels:
            p_now = prices.loc[disc_prices==aLevel]
            p_chooser = (disc_prices==aLevel).shift(1)
            p_chooser[0] = (disc_prices==aLevel).iloc[-1]
            p_next = prices.loc[p_chooser]
            p_diff = pd.Series(p_next.values - p_now.values)
            self.k_samplers.append(p_diff + aLevel)
    
    def fitted_value_iteration_k_sampling(self, prices_train, nSamples = 100, gamma = 0.96, maxIter = 100, K=50):
        y = prices_train.values
        
        n = nSamples
        t_samples = [random.randint(0,len(y)-1) for i in range(n)]
        samples = [[random.randint(0,5)*100, y[t_samples[i]]] for i in range(n)]
        theta = np.zeros(len(samples[0]))
        prev_theta = np.zeros(len(samples[0]))
        yy = np.zeros(n)
        iters = 0
        
        while True:
            iters += 1
            if iters > maxIter: break
            #print("Iteration: ", iters)
            for i in range(n):
                q = np.zeros(len(self.actions))
                s = samples[i][0]
                p = samples[i][1]
                level = np.argmin(abs(self.price_levels - p))
                for ix, a in enumerate(self.actions):
                    new_s = s+a
                    if new_s > 500 or new_s < 0:
                        R = -np.inf
                    else:
                        R = - a * p
                    s_prime = []
                    sampler = self.k_samplers[level]
                    
                    for k in range(K):
                        s_prime.append([s + a, y[random.randint(0,len(sampler)-1)]])
                    q[ix] = R + sum(gamma * (self.phi(s_prime) @ theta))/K
                yy[i] = max(q)
            theta = self.cf_fit(self.phi(samples),yy)
            if all(theta == prev_theta):
                break
            prev_theta = theta
        self.theta = theta
        
        
        return iters
    
    def plot_test_results(self, profit_list = None):
        if profit_list is not None:
            self.test_profits = profit_list
        
        plt.title(str("\'Continuous\' pricing: "))
        plt.plot(np.array(self.test_profits))
        #plt.plot(self.socs)
        plt.ylabel("Total profit")
        plt.xlabel("Hour of trading in test market")
        plt.show()
    
    def test(self, p_test, plot = False, soc = 200, profit = 0):
        if self.theta.any() == None:
            print("Need to run fitted_value_iteration first to define theta")
        else:
            profits = [profit]
            socs = [soc]
            
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

    