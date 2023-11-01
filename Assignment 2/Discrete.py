import os
import pandas as pd
import numpy as np

def MDP(rewards, probs, gamma,maxIter):
    last_val = np.zeros(rewards.shape)
    values = np.zeros(rewards.shape)
    iters = 0
    while True:
        for p in range(len(values)):
            for s in range(len(values[p])):
                for a in range(len(values[p,s])):
                    # Calculating sum
                    val_sum = 0
                    # Either make cost of going to impossible soc infinite or
                    # Limit charge/discharge at edge state
                    for i in range(len(probs[p,s])):
                        print(p,s,a,i)
                        if gamma*probs[p,s,i] != 0:
                            val_sum += gamma*probs[p,s,i]*max(values[p+i-1,s+a-1,[0,1,2]])
                    
                    values[p,s,a] = rewards[p,s,a] + val_sum
        
        if values==last_val or iters >= maxIter:
            break
        last_val = values
        iters += 1
    
    return(values)
        

cwd = os.getcwd()
temp_dir = os.path.join(cwd, 'Prices.csv')
df = pd.read_csv(temp_dir)

prices = df.Spot.values

rewards = np.zeros((3,8,3))
for i in range(len(rewards)):
    for j in range(len(rewards[i])):
        rewards[i,j,0] = 1
        rewards[i,j,1] = 0
        rewards[i,j,2] = -1
rewards[:,0,0] = -100
rewards[:,7,2] = -100

#values = np.zeros((3,6,3))
probs = np.zeros((3,8,3))
for i in range(len(probs)):
    for j in range(len(probs[i])):
        for k in range(len(probs[i,j])):
            probs[i,j,k] = 1/3

probs[0]=0.5
probs[0,:,0]=0
probs[2]=0.5
probs[2,:,2]=0


gamma  = 0.9


values = MDP(rewards=rewards,probs=probs,gamma=gamma,maxIter=1000)
# For every state have three actions, -1, 0, 1
# while True:
#     for p in range(len(values)):
#         for s in range(len(values[p])):
#             for a in range(len(values[p,s])):
#                 # Calculating sum
#                 val_sum = 0
#                 # Either make cost of going to impossible soc infinite or
#                 # Limit charge/discharge at edge state
#                 for i in range(len(probs[p,s])):
#                     val_sum += gamma*probs[p,s,i]*max(values[p+i-1,s+a-1,[0,1,2]])
                    
#                 values[p,s,a] = rewards[p,s,a] + val_sum
                

