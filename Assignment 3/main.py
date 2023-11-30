
import unit_commitment as uc
import sample_gen as sg
import numpy as np
import pandas as pd
import os
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import plot
import matplotlib.pyplot as plt
import random

#%% Step 3.1: Create samples
cwd = os.getcwd()
demand = pd.read_csv(cwd + '/system_data/demand.csv')
demand = pd.DataFrame(demand)
demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])

""" Task: More samples and shuffle the way each day is generated """
""" Task: Create samples which activates line constraints """
n_samples = 200

#multis = np.linspace(0.5, 1.5, n_samples)
#demands = [demand*multi for multi in multis]

demands = sg.generate_samples(n_samples)

#%% Step 3: Create data sets 
X = []
y_G = []
y_L = []

for i, sample in enumerate(demands.items()):
    print("Sample: ", i, ". Scaling: ", sample[0])
    X_sample, y_G_sample, y_L_sample, _, _ = uc.uc(sample[1], log=True)
    X.append(X_sample)
    y_G.append(y_G_sample)
    y_L.append(y_L_sample)

n_samples = len(X)

X_model = np.array(X).reshape(24*n_samples,91)
y_G_model = np.array(y_G).reshape(54,24*n_samples)
y_L_model = np.array(y_L).reshape(186,24*n_samples)

#%% Save sample training and test data for future quick runs/reruns
X_file = cwd + "/generated_samples/X.csv"
y_G_file = cwd + "/generated_samples/y_G.csv"
y_L_file = cwd + "/generated_samples/y_L.csv"

pd.DataFrame(X_model).to_csv(X_file)
pd.DataFrame(y_G_model).to_csv(y_G_file)
pd.DataFrame(y_L_model).to_csv(y_L_file)

#%% Load sample training and test data
X_file = cwd + "/generated_samples/X_model.csv"
y_G_file = cwd + "/generated_samples/y_G_model.csv"
y_L_file = cwd + "/generated_samples/y_L_model.csv"
X_model = np.asarray(pd.read_csv(X_file,index_col=0))
y_G_model = np.asarray(pd.read_csv(y_G_file,index_col=0)).astype(int)
y_L_model = np.asarray(pd.read_csv(y_L_file,index_col=0)).astype(int)

plt.scatter(np.arange(len(y_G_model))+1,y_G_model.sum(axis=1),label = "Active generator hours",s=10,marker="_")
plt.title("Number of hours each generator is on")
plt.ylabel("Hours")
plt.xlabel("Generators")
plt.grid(axis='y')
plt.ylim(0,1000)
plt.xlim(0,len(y_G_model)+1)
plt.legend()
plt.show()

plt.scatter(np.arange(len(y_L_model))+1,y_L_model.sum(axis=1),label = "Active constraint hours",marker="o")
plt.title("Number of hours each line constraint is active")
plt.ylabel("Hours")
plt.xlabel("Lines")
plt.grid(axis='y')
plt.ylim(-1,6)
plt.xlim(0,len(y_L_model)+1)
plt.legend()
plt.show()

#%% Create altered data set 
X_1hours = X_model.copy()

# Initializing lists
X_3hours = list()

# Create sequences of 3 hours for X
for h in range(len(X_model)):
    if h % 24 == 0:
        triple = np.concatenate([X_1hours[h], X_1hours[h], X_1hours[h+1]])
    elif h % 24 == 23:
        triple = np.concatenate([X_1hours[h-1], X_1hours[h], X_1hours[h]])
    else:
        triple = X_model[h-1:h+2].flatten()
    X_3hours.append(triple)

# Change the type 
X_3hours = np.asarray(X_3hours)

#%% Number of days and creating 3 slices of approx. 60%, 20%, and 20% size.
n_hours = int(X_model.shape[0])
n_days = int(n_hours/24)
daily_ids = np.arange(n_days)
rand_seed = 42
random.Random(rand_seed).shuffle(daily_ids)
hourly_ids = []
for i in daily_ids:
    hours = list(np.arange(i*24,(i+1)*24))
    hourly_ids.extend(hours)

# Days in each set of data
train_slice_days = daily_ids[:int(0.6*n_days)]
val_slice_days = daily_ids[int(0.6*n_days):int(0.8*n_days)]
# Days to compare performance of solver with after classification:
test_slice_days = daily_ids[int(0.8*n_days):]

# Slices to be used:
train_slice = hourly_ids[:int(0.6*n_days)*24]
val_slice = hourly_ids[int(0.6*n_days)*24:int(0.8*n_days)*24]
test_slice = hourly_ids[int(0.8*n_days)*24:]
train_val_slice = hourly_ids[:int(0.8*n_days)*24]

#%% Train-val-test splits:
X_1_train = X_1hours[train_slice]
X_1_val = X_1hours[val_slice]
X_1_test = X_1hours[test_slice]
X_1_train_full = X_1hours[train_val_slice]

X_3_train = X_3hours[train_slice]
X_3_val = X_3hours[val_slice]
X_3_test = X_3hours[test_slice]
X_3_train_full = X_3hours[train_val_slice]

y_G_train = y_G_model.T[train_slice].T
y_G_val = y_G_model.T[val_slice].T
y_G_test = y_G_model.T[test_slice].T
y_G_train_full = y_G_model.T[train_val_slice].T

y_L_train = y_L_model.T[train_slice].T
y_L_val = y_L_model.T[val_slice].T
y_L_test = y_L_model.T[test_slice].T
y_L_train_full = y_L_model.T[train_val_slice].T

#%% Perform upsampling
def upsample(X_train, y_train):
    X_ones = X_train[y_train.sum(axis=0) >= 1]
    y_ones = np.transpose(y_train)[y_train.sum(axis=0) >= 1]
    n_samples = len(X_train) - 2*len(X_ones)
    X_ones_upsampled = resample(X_ones, random_state=42, n_samples=n_samples, replace=True)
    y_ones_upsampled = resample(y_ones, random_state=42, n_samples=n_samples, replace=True)

    return np.append(X_train, X_ones_upsampled, axis=0), np.append(y_train, np.transpose(y_ones_upsampled), axis=1)

X_1_train_up, y_L_train_up = upsample(X_1_train, y_L_train)
X_3_train_up, y_L_train_up = upsample(X_3_train, y_L_train)

#%% Step 4: Classification

def classifiers(X_train, y_train, X_test, y_test, clfs=[svm.SVC()], target="G"):

    test_accuracies = [{} for clf in clfs]
    train_accuracies = [{} for clf in clfs]
    classifiers = [{} for clf in clfs]
    predictions = [{} for clf in clfs]
    
    for g in range(len(y_train)):
        for i in range(len(clfs)):
            clf = clfs[i]
            if np.sum(y_train[g]) == 0:
                y_pred = np.zeros(len(X_test))
                y_pred_train = np.zeros(len(X_train))
                print("Boink", g)
            else:
                clf.fit(X_train, y_train[g].astype(int))
                y_pred = clf.predict(X_test)
                y_pred_train = clf.predict(X_train)
            accuracy = accuracy_score(y_test[g],y_pred)
            accuracy_train = accuracy_score(y_train[g],y_pred_train)
            test_accuracies[i][str(target + ":" + str(g))] = accuracy
            train_accuracies[i][str(target + ":" + str(g))] = accuracy_train
            classifiers[i][str(target + ":" + str(g))] = clf
            predictions[i][str(target + ":" + str(g))] = y_pred
    
    return test_accuracies, train_accuracies, classifiers, predictions

#%% Tuning regularization hyperparameter for the two models
def tuneRegu(model_type="linear", C_list=np.linspace(0.1,1,10)):
    accs = []
    accs_train = []
    for C in C_list:
        print(C)
        if model_type == "linear":
            clfs = [svm.LinearSVC(C=C, dual="auto")]
        elif model_type == "non-linear":
            clfs = [svm.SVC(C=C)]
        acc, acc_train, _, _ = classifiers(X_1_train[:24*10], y_G_train[:,:24*10], X_1_val[:24*5], y_G_val[:,:24*5], clfs, target="G")
        accs.append(np.mean(list(acc[0].values())))
        accs_train.append(np.mean(list(acc_train[0].values())))

    best_ix = np.argmax(accs)
    best_C = C_list[best_ix]

    return accs, best_C, accs_train

#%% Validation of optimal regularization parameter
accs_val_lin, C_lin, accs_train_lin = tuneRegu(model_type="linear", C_list=np.linspace(0.0001,1,3)) #C_list=np.logspace(-5,3,9)

plt.plot(accs_val_lin, label="Validation accuracies")
plt.plot(accs_train_lin, label="Training accuracies")
plt.xscale('log')
plt.axvline(np.argmax(accs_val_lin), C_lin, linestyle="--",label="Optimal regularization")
plt.legend()
plt.title("Regularization of the linear SVM")
plt.show()

#%%
accs_val_non, C_non, accs_train_non = tuneRegu(model_type="non-linear", C_list=np.linspace(1,100,3)) #C_list=np.logspace(-5,3,9)

plt.plot(accs_val_non, label="Validation accuracies")
plt.plot(accs_train_non, label="Training accuracies")
plt.xscale('log')
plt.axvline(np.argmax(accs_val_non),C_non, linestyle="--",label="Optimal regularization")
plt.legend()
plt.title("Regularization of the non-linear SVM")
plt.show()

#%% Comparing performance of classifiers
# Classifiers to compare
clfs = [svm.LinearSVC(dual="auto", C=C_lin), svm.SVC(C=C_non), RandomForestClassifier()]
# clfs = [svm.NuSVC(gamma="auto"), RandomForestClassifier()]

### Predict generator status ###
acc_1_G, _, clf_1_G, y_1_pred_G = classifiers(X_1_train, y_G_train, X_1_val, y_G_val, clfs, target="G")

### Step 4 - Predict active constraints ###
acc_1_L, _, clf_1_L, y_1_pred_L = classifiers(X_1_train, y_L_train, X_1_val, y_L_val, clfs, target="L")

### Predict generator status ###
acc_3_G, _, clf_3_G, y_3_pred_G = classifiers(X_3_train, y_G_train, X_3_val, y_G_val, clfs, target="G")

### Step 4 - Predict active constraints ###
acc_3_L, _, clf_3_L, y_3_pred_L = classifiers(X_3_train, y_L_train, X_3_val, y_L_val, clfs, target="L")

""" Task: Other classifiers? SVM always predicts 0 currently,
    so something definitely needs to change - or is that the conclusion for
    our linear solver? """


#%% Evaluation of model accuracies:
plot.accComparison(acc_1_G, target="G", hours=1, models=["Lin SVM", "Non-lin SVM", "RF"])
plot.accComparison(acc_1_L, target="L", hours=1, models=["Lin SVM", "Non-lin SVM", "RF"])

plot.accComparison(acc_3_G, target="G", hours=3, models=["Lin SVM", "Non-lin SVM", "RF"])
plot.accComparison(acc_3_L, target="L", hours=3, models=["Lin SVM", "Non-lin SVM", "RF"])


# Evaluation of unit commitment feasibility:
""" Task: Evaluation should also be in terms of actually performing the
    unit commitment. So use the predicted values of 24 subsequent hours
    and see if it is feasible maybe. """

# Maybe plot potential relation between cost of gen and accuracy of prediction

#%% Confusion matrices
plot.cf_matrix(y_G_val.flatten(), np.asarray(list(y_1_pred_G[0].values())).flatten(), title="(generator, 1 hour, Lin SVM)")
plot.cf_matrix(y_G_val.flatten(), np.asarray(list(y_1_pred_G[1].values())).flatten(), title="(generator, 1 hour, Non-lin SVM)")
plot.cf_matrix(y_G_val.flatten(), np.asarray(list(y_1_pred_G[2].values())).flatten(), title="(generator, 1 hour, RF)")

plot.cf_matrix(y_L_val.flatten(), np.asarray(list(y_1_pred_L[0].values())).flatten(), title="(line, 1 hour, Lin SVM)")
plot.cf_matrix(y_L_val.flatten(), np.asarray(list(y_1_pred_L[1].values())).flatten(), title="(line, 1 hour, Non-lin SVM)")
plot.cf_matrix(y_L_val.flatten(), np.asarray(list(y_1_pred_L[2].values())).flatten(), title="(line, 1 hour, RF)")

plot.cf_matrix(y_G_val.flatten(), np.asarray(list(y_3_pred_G[0].values())).flatten(), title="(generator, 3 hour, Lin SVM)")
plot.cf_matrix(y_G_val.flatten(), np.asarray(list(y_3_pred_G[1].values())).flatten(), title="(generator, 3 hour, Non-lin SVM)")
plot.cf_matrix(y_G_val.flatten(), np.asarray(list(y_3_pred_G[2].values())).flatten(), title="(generator, 3 hour, RF)")

plot.cf_matrix(y_L_val.flatten(), np.asarray(list(y_3_pred_L[0].values())).flatten(), title="(line, 3 hour, Lin SVM)")
plot.cf_matrix(y_L_val.flatten(), np.asarray(list(y_3_pred_L[1].values())).flatten(), title="(line, 3 hour, Non-lin SVM)")
plot.cf_matrix(y_L_val.flatten(), np.asarray(list(y_3_pred_L[2].values())).flatten(), title="(line, 3 hour, RF)")


#%% Best model for each prediction task
C_lin = 1e-4
best_model_G = svm.LinearSVC(dual="auto", C=C_lin)
acc_G, _, _, y_pred_test_G = classifiers(X_3_train_full, y_G_train_full, X_3_test, y_G_test, [best_model_G], target="G")

best_model_L = svm.LinearSVC(dual="auto", C=C_lin)
acc_L, _, _, y_pred_test_L = classifiers(X_3_train_full, y_L_train_full, X_3_test, y_L_test, [best_model_L], target="L")


#%% Step 6
n_test_days = len(test_slice_days)

#test_day = 0
#test_day = max(min([test_day, n_test_days-1]),0)
non_opts = 0
T1 = []
T2 = []
T3 = []
for test_day in range(n_test_days):
    hours = test_slice[test_day*24:(test_day+1)*24]
    demand_test = X_1_test[test_day*24:(test_day+1)*24]
    
    pred_G_test = np.asarray([values[24*test_day:24*(test_day+1)] for values in y_pred_test_G[0].values()])
    pred_L_test = np.asarray([values[24*test_day:24*(test_day+1)] for values in y_pred_test_L[0].values()]).T
    
    _, _, _, opt_real, t_no_init = uc.uc(demand_test, log=False)
    print("Time to solve without any initialization:    ", t_no_init)
    
    """ Task: Evaluation of runtime of unit commitment problem when initialing
        with predicted 'b'-variables. """
    
    _, _, _, opt_g, t_with_init = uc.uc(demand_test, b_pred = pred_G_test, log=False)
    print("Time to solve with generator initialization: ", t_with_init)
    T1.append(t_no_init-t_with_init)
    """ Task: Evaluation of runtime of unit commitment problem when trimming
        problem by removing predicted inactive line constraints. """    
    _, _, _, opt_l, t_trim = uc.uc(demand_test, active_lines = pred_L_test, log=False)
    print("Time to solve with constraint trimming:      ", t_trim)
    T2.append(t_no_init-t_trim)
    # All predictions utilised:
    _, _, _, opt_all, t_all = uc.uc(demand_test, b_pred = pred_G_test, active_lines = pred_L_test, log=False)
    print("Time to solve with both model alterations:   ", t_all)
    T3.append(t_no_init-t_all)
    
    print()
    print("Actual optimum:                        ", opt_real)
    print("Optimum with trimmed constraints:      ", opt_l)
    if (opt_real-opt_l)/opt_real > 0.001:
        print("Not same solution")
        non_opts += 1

print("Average time saved with gen inits: ", np.median(T1))
print("Average time saved with constraint trimming: ", np.median(T2))
print("Average time saved with both implemented: ", np.median(T3))
print("No. of days where the uc was sub-optimal/infeasible: ", non_opts)

#%% Step 7 - plotting of PCA
C_non = 10
plot.pca_contour_plot(C_non, X_1_train, X_1_test, y_G_test[53], title='non-linear SVM on PCA; not-regularized', linear=False)
C_non2 = 0.1
plot.pca_contour_plot(C_non2, X_1_train, X_1_test, y_G_test[53],title='non-linear SVM on PCA; regularized', linear=False)
C_lin=1e-4
plot.pca_contour_plot(C_lin, X_1_train, X_1_test, y_G_test[53], title='linear SVM on PCA', linear=True)

#plot.pca_plot(X_3_train_full, X_3_test, y_pred_test_G[0], gen='G:53', pc = [1,2])
#plot.pca_plot(X_3_train_full, X_3_test, {'G:53':y_G_test[0]}, gen='G:53', pc = [1,2])
#plot.pca_plot(X_3_train_full, X_3_train_full, y_pred_G_G[0], gen='G:31', pc = [1,2])
#plot.pca_plot(X_3_train_full, X_3_train_full, {'G:31':y_G_train_full[0]}, gen='G:31', pc = [1,2])
#plot.pca_plot(X_train, X_test, y_pred_G_svm, gen='G:37', pc = [1,2])

#%% PCA confusion matrices
pca = PCA(n_components=2)
pca.fit(X_1_train)
reduced_X_train  = pca.transform(X_1_train)
reduced_X = pca.transform(X_1_test)
PCA_clfs = [svm.LinearSVC(dual="auto", C=C_lin), svm.SVC(C=C_non), svm.SVC(C=C_non2)]
y_pred_PCA = []
for i in range(len(PCA_clfs)):
    clf = PCA_clfs[i]
    clf.fit(reduced_X_train, y_G_train[53])
    y_pred_PCA.append(clf.predict(reduced_X))

plot.cf_matrix(y_G_test[53].flatten(), np.asarray(list(y_pred_PCA[0])).flatten(), title="(gen 54, Lin SVM, PCA)")
plot.cf_matrix(y_G_test[53].flatten(), np.asarray(list(y_pred_PCA[1])).flatten(), title="(gen 54, non-Lin SVM, PCA)")
plot.cf_matrix(y_G_test[53].flatten(), np.asarray(list(y_pred_PCA[2])).flatten(), title="(gen 54, non-Lin SVM, reg, PCA)")




