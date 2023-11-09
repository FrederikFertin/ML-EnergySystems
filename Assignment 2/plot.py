import matplotlib.pyplot as plt

def plotCompareProfits(profits1, profits2=None, profits3=None, p_test=[], labels=None, title=''):
    
    fig, ax1 = plt.subplots()

    color = ['tab:red', 'tab:blue', 'tab:green']
    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('Cumulated profits')
    ax1.plot(profits1, color=color[0], label=labels[0])
    if (profits2 != None):
        ax1.plot(profits2, color=color[1], label=labels[1])
    if (profits3 != None):
        ax1.plot(profits3, color=color[2], label=labels[2])
    
    if len(p_test) > 0:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
        color = 'tab:grey'
        ax2.set_ylabel('Prices', color=color)  # we already handled the x-label with ax1
        ax2.plot(p_test.values, color=color, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


