import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.decomposition import PCA

def accComparison(accuracies, target="G", hours=1, models=["","",""]):
    colors = ['blue','green', 'red']
    
    # x1 = x-0.2
    # x2 = x+0.2
    i = 0
    for acc in accuracies:
        mean = np.mean(list(acc.values()))
        x = np.arange(1,len(acc.values())+1)
        plt.scatter(x, acc.values(), label=models[i], color=colors[i])
        plt.axhline(mean,label="Mean "+models[i],color=colors[i], alpha=0.5, linestyle = "--")
        i += 1
    # plt.scatter(x, svm_accuracies.values(), label="SVM",color=colors[1])
    # plt.axhline(mean_svm,label="Mean SVM",color=colors[1], alpha=0.5, linestyle = "--")
    if target == "G":
        plt.xlabel("Generator")
        plt.title("Binary classifier accuracies for each generator (hours=" + str(hours) + ")")
    elif target == "L":
        plt.xlabel("Line")
        plt.title("Binary classifier accuracies for each line (hours=" + str(hours) + ")")
    else: 
        raise KeyError("Only 'G' and 'L' are accepted for arg. 'target'")
    plt.grid(True)
    ticks = np.arange(3,len(x),3)
    plt.xticks(ticks, ticks)
    plt.ylabel("Accuracy [-]")
    plt.legend()
    plt.show()
    # plt.plot(y_model.sum(axis=1),label = "Active generator hours")
    # plt.legend()
    # plt.show()

def cf_matrix(y_true, y_pred, title=""):
    
    cf_M = confusion_matrix(y_true, y_pred)
    
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_M.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_M.flatten()/np.sum(cf_M)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_M, annot=labels, fmt='', cmap='Blues', annot_kws={"size": 20})
    plt.title(title, fontsize=20)
    plt.xlabel("Predicted label", fontsize=15)
    plt.ylabel("True label", fontsize=15)
    plt.show()

def pca_plot(X_train, X_test, y_pred, gen = 'G:31', pc = [1,2]):
    labels = y_pred[gen]
    pca = PCA(n_components=max(pc))
    pca.fit(X_train)
    reduced_X = pca.transform(X_test)
    plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.set_facecolor("lightgrey")
    plt.scatter(reduced_X[:, pc[0]-1], reduced_X[:, pc[1]-1], c=labels, cmap='coolwarm', marker='o', s=8)
    #plt.title('Binary classification - Principal components')
    plt.title("Predictions",fontsize=30)
    plt.xlabel('PC' + str(pc[0]))
    plt.ylabel('PC' + str(pc[1]))
    plt.colorbar(label='Classification')
    plt.show()