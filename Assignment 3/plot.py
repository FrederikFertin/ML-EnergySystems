import matplotlib.pyplot as plt
import numpy as np

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

