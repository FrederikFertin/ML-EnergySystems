import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def accComparison(svm_accuracies, rf_accuracies):
    mean_svm = np.mean(list(svm_accuracies.values()))
    mean_rf = np.mean(list(rf_accuracies.values()))
    colors = ['blue','green']

    plt.plot(rf_accuracies.values(), label="RF",color=colors[0])
    plt.axhline(mean_rf,label="Mean RF",color=colors[0], alpha=0.5, linestyle = "--")
    plt.plot(svm_accuracies.values(), label="SVM",color=colors[1])
    plt.axhline(mean_svm,label="Mean SVM",color=colors[1], alpha=0.5, linestyle = "--")
    plt.legend()
    plt.title("Binary classifier accuracies for each generator")
    plt.show()
    # plt.plot(y_model.sum(axis=1),label = "Active generator hours")
    # plt.legend()
    # plt.show()

def pca_plot(X_train, X_test, y_pred, gen = 'G:31', pc = [1,2]):
    labels = y_pred[gen]
    pca = PCA(n_components=max(pc))
    pca.fit(X_train)
    reduced_X = pca.transform(X_test)
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_X[:, pc[0]-1], reduced_X[:, pc[1]-1], c=labels, cmap='viridis', marker='o')
    plt.title('Binary classification - Principal components')
    plt.xlabel('PC' + str(pc[0]))
    plt.ylabel('PC' + str(pc[1]))
    plt.colorbar(label='Classification')
    plt.show()