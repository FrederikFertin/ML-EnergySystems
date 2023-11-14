import matplotlib.pyplot as plt
import numpy as np

def plotPriceData(prices, p_cuts, train_length=180, test_length=50):
    t = np.arange(len(prices))
    plt.figure()
    plt.scatter(t, prices, alpha=0.3, s=0.5)
    plt.plot(t, np.ones(len(t))*p_cuts[1], linestyle='--', color='black')
    plt.plot(t, np.ones(len(t))*p_cuts[2], linestyle='--', color='black')
    
    train1_end = (train_length*24)/len(t)
    test1_end = train1_end + (test_length*24)/len(t)
    train2_end = (8760 + train_length*24)/len(t)
    test2_end = train2_end + (test_length*24)/len(t)
    
    plt.axhspan(prices.min(), prices.max(), 0, train1_end, color='grey', alpha=0.3)
    plt.axhspan(prices.min(), prices.max(), train1_end, test1_end, color='grey', alpha=0.7)
    plt.axhspan(prices.min(), prices.max(), (8760/len(t)), train2_end, color='grey', alpha=0.3)
    plt.axhspan(prices.min(), prices.max(), train2_end, test2_end, color='grey', alpha=0.7)
    plt.xlim(t[0], t[-1])
    plt.ylim(prices.min(),prices.max())
    plt.xlabel("Hour")
    plt.ylabel("price [€]")
    plt.title("Spot prices in 2021 and 2022")
    plt.show()
    
def plotThetaConvergence(thetas):
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
    
def plotCompareProfits(profits, p_test=[], labels=None, title=''):
    
    fig, ax1 = plt.subplots()

    # color = ['tab:red', 'tab:blue', 'tab:green']
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('Cumulated profits')
    for i in range(len(profits)):    
        ax1.plot(profits[i], label=labels[i], linewidth=0.7)
    
    if len(p_test) > 0:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
        color = 'tab:grey'
        ax2.set_ylabel('Prices', color=color)  # we already handled the x-label with ax1
        ax2.plot(p_test.values, color=color, linewidth=0.7, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title)
    ax1.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


