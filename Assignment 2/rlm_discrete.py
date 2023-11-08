import numpy as np

class RLM_discrete:    
    
    def __init__(self, plevels=[0,1,2], p_cuts=[0.5,1.5], socs=np.linspace(0,500,6), actions=np.array([-100,0,100])):
        """
        Initialises state and action space as well as a uniform probability 
        matrix and a reward matrix. 
        
        Args:
            plevels: list of price levels. 
            p_cuts: list of prices to differentiate into price levels. 
            socs: discrete values which the soc can attain. 
            actions: np.array defining the action space. 
        """
        
        self.socs = socs
        self.n_socs = len(self.socs)
        self.actions = actions
        self.n_actions = len(self.actions)
        self.price_cuts = p_cuts
        self.setPriceLevels(plevels)
        self.P = np.ones(
            (self.n_levels,self.n_levels)
             ) / self.n_levels
        self.calcRmatrix()
    
    def setPriceLevels(self, plevels):
        """
        Changes the price levels of the instance. 
        
        Args:
            plevels: list of price levels. 
        """
        
        self.price_levels = plevels
        self.n_levels = len(self.price_levels)
    
    def displayInfo(self):
        """
        Prints the socs and actions space of the instance. 
        """
        
        print(self.socs)
        print(self.actions)
    
    def calcPmatrix(self, prices):
        """
        Learns the state transition probability matrix from historical price
        data by counting all occurences of each transition and dividing by 
        all occurences of the from state. The probability matrix has 
        dimensions #price levels x #price levels. 
        
        Args: 
            prices: pd.DataFrame object containing historical prices. 
            price_levels: 
        """
        
        prices = np.round(prices,2)
        price_levels = np.round(self.price_levels,2)
        
        C = np.zeros((self.n_levels,self.n_levels))
        price_level_count = np.zeros(self.n_levels)
        last_price = prices.iloc[-1]
        for (ix, current_price) in enumerate(prices):
            if ix == 0:
                current_price = prices.iloc[0]
            from_price = np.where(np.array(price_levels) == last_price)[0][0]
            to_price = np.where(np.array(price_levels) == current_price)[0][0]
            C[from_price,to_price] += 1
            last_price = current_price
            price_level_count[from_price] +=1
        
        #self.P = C/len(prices)*len(price_levels)
        self.P = C/price_level_count
    
    def calcProbability(self, s1, a, s2):
        """
        Uses the probability matrix self.P to calculate the probability 
        of going from one state to another given a certain action. 
        
        Args: 
            s1: list of state components for "current state" [price, SOC]
            a: action, i.e., charging rate (-100, 0 or 100)
            s2: list of state components for "next state" [price, SOC]
        
        Returns:
            Probability of going from s1 to s2 given action a.
        """
        
        if s1[1] + a == s2[1]:
            return self.P[s1[0]+1, s2[0]+1]
        else:
            return 0
    
    def calcRmatrix(self):
        """
        Calculates the reward matrix by looping through all combinations of 
        states and actions and setting the reward equal to:
            
            R = - price level * action 
            
        The reward of discharging at SOC = 0 and charging at SOC = 500 MWh is 
        set to -infinity. 
        """
        
        self.R = np.zeros(
            (self.n_levels, self.n_socs, self.n_actions)
            )
        
        for (l,level) in enumerate(self.R):
            for (s, soc) in enumerate(level):
                for (a, action) in enumerate(soc):
                    if (s == 0 and a == 0)\
                        or (s == self.n_socs-1 and a == self.n_actions-1):
                        self.R[l,s,a] = - np.inf
                    else:
                        self.R[l,s,a] = - self.price_levels[l] * self.actions[a]

    def valueIter(self, gamma = 0.9, maxIter = 1000):
        """
        Performs the value iteration algorithm. 
        
        Args: 
            gamma: discount factor 
            maxIter: maximum number of iterations before termination in case 
                     convergence is not achieved. 
        
        Returns:
            values: Matrix containing values of all states. 
            iters: Number of iterations performed. 
        
        """
        
        last_val = np.zeros((self.n_levels, self.n_socs))
        values = np.zeros((self.n_levels, self.n_socs))
        policy = np.zeros((self.n_levels, self.n_socs))
        iters = 0
        while True:
            for p in range(self.n_levels):
                for s in range(self.n_socs):
                    vals = []
                    for a in range(self.n_actions):
                        # Calculating sum
                        val_sum = self.R[p,s,a]
                        # Either make cost of going to impossible soc infinite or
                        # Limit charge/discharge at edge state
                        for i in range(len(self.P[p])):
                            if s+a-1 >= 0 and s+a-1 < self.n_socs:
                                val_sum += gamma * self.P[p,i] * last_val[i,s+a-1]
    
                        vals.append(val_sum)
                    values[p,s] = max(vals)
                    policy[p,s] = self.actions[np.argmax(vals)]
                    
            if np.all(values==last_val) or iters >= maxIter:
                break
            last_val = values.copy()
            iters += 1
            #print('test')
        self.V = values
        self.Pi = policy
        
        return values, iters
    
    def test(self, p_test):
        """
        Tests performance of the model (i.e., the learned policy) by 
        calculating the generated profits when trading in the market given some 
        future prices. 
        
        Args:
            p_test: pd.DataFrame containing future electricity prices.
            
        Returns:
            profits: list of accumulated profits during trading period. 
            socs: SOC during trading period. 
        """
        
        prices = p_test.copy()
        profits = [0]
        socs = [200]
        
        d_prices = p_test.copy()
        for i in range(self.n_levels):
            if i == 0:
                d_prices.loc[(prices < self.price_cuts[i+1])] = self.price_levels[i]
            elif i < self.n_levels-1:
                d_prices.loc[(prices >= self.price_cuts[i]) & (prices < self.price_cuts[i+1])] = self.price_levels[i]
            else:
                d_prices.loc[prices >= self.price_cuts[i]] = self.price_levels[i]
        
        for t in range(len(prices)):
            p = np.where(np.round(self.price_levels,2) == np.round(d_prices.iloc[t],2))[0][0]
            
            s = np.where(self.socs == socs[-1])[0][0]
            
            a = self.Pi[p,s]
            price = prices.iloc[t]
            profits.append(profits[-1] + price * (-a))
            socs.append(socs[-1] + a)
        
        return profits, socs
