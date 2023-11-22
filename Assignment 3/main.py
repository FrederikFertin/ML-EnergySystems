
import unit_commitment as uc
import sample_gen as sg
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import plot
import matplotlib.pyplot as plt

#%% Step 3.1: Create samples
cwd = os.getcwd()
demand = pd.read_csv(cwd + '/system_data/demand.csv')
demand = pd.DataFrame(demand)
demand = np.vstack([list(map(float, row[0].split(';'))) for row in demand.values])

""" Task: More samples and shuffle the way each day is generated """
""" Task: Create samples which activates line constraints """
n_samples = 20

multis = np.linspace(1.6, 1.65, n_samples)

demands = [demand*multi for multi in multis]

# = sg.generate_samples(multipliers=multis)

#%% Step 3: Create data sets 
X = []
y_G = []
y_L = []

for i, sample in enumerate(demands):
    print("Sample ", i)
    X_sample, y_G_sample, y_L_sample, _, _ = uc.uc(sample, log=False)
    X.append(X_sample)
    y_G.append(y_G_sample)
    y_L.append(y_L_sample)

X_model = np.array(X).reshape(24*n_samples,91)
y_G_model = np.array(y_G).reshape(54,24*n_samples)
y_L_model = np.array(y_L).reshape(186,24*n_samples)

plt.plot(y_G_model.sum(axis=1),label = "Active generator hours")
plt.legend()
plt.show()

plt.plot(y_L_model.sum(axis=1),label = "Active constraints")
plt.legend()
plt.show()

#%% Save sample training and test data for future quick runs/reruns
X_file = cwd + "/generated_samples/X_model.csv"
y_G_file = cwd + "/generated_samples/y_G_model.csv"
y_L_file = cwd + "/generated_samples/y_L_model.csv"

#%% 
# pd.DataFrame(X_model).to_csv(X_file)
# pd.DataFrame(y_G_model).to_csv(y_G_file)
# pd.DataFrame(y_L_model).to_csv(y_L_file)

#%% Load sample training and test data
X_model = np.asarray(pd.read_csv(X_file,index_col=0))
y_G_model = np.asarray(pd.read_csv(y_G_file,index_col=0))
y_L_model = np.asarray(pd.read_csv(y_L_file,index_col=0))

#%% Create altered data set 
X_1hours = X_model.copy()

# Initializing lists 
X_3hours = list()

# Create sequences of 3 hours for X
for h in range(1,len(X_model)-1):
    X_3hours.append(X_model[h-1:h+2].flatten())

# Change the type 
X_3hours = np.asarray(X_3hours)

#%% Create train, validation, and test sets 
# 1 hour data set
# Test splits
X_1_temp, X_1_test, y_1_G_temp, y_1_G_test = train_test_split(X_1hours, np.transpose(y_G_model), test_size=0.2, shuffle=True, random_state=42)
_, _, y_1_L_temp, y_1_L_test = train_test_split(X_1hours, np.transpose(y_L_model), test_size=0.2, shuffle=True, random_state=42)    

# Validation splits
X_1_train, X_1_val, y_1_G_train, y_1_G_val = train_test_split(X_1_temp, y_1_G_temp, test_size=0.25, shuffle=True, random_state=42)
_, _, y_1_L_train, y_1_L_val = train_test_split(X_1_temp, y_1_L_temp, test_size=0.25, shuffle=True, random_state=42)

y_1_G_train, y_1_G_val, y_1_G_test = np.transpose(y_1_G_train), np.transpose(y_1_G_val), np.transpose(y_1_G_test)
y_1_L_train, y_1_L_val, y_1_L_test = np.transpose(y_1_L_train), np.transpose(y_1_L_val), np.transpose(y_1_L_test)

# 3 hours in each data point 
# Test splits
X_3_temp, X_3_test, y_3_G_temp, y_3_G_test = train_test_split(X_3hours, np.transpose(y_G_model)[1:-1], test_size=0.2, shuffle=True, random_state=42)
_, _, y_3_L_temp, y_3_L_test = train_test_split(X_3hours, np.transpose(y_L_model)[1:-1], test_size=0.2, shuffle=True, random_state=42)    

# Validation splits
X_3_train, X_3_val, y_3_G_train, y_3_G_val = train_test_split(X_3_temp, y_3_G_temp, test_size=0.25, shuffle=True, random_state=42)
_, _, y_3_L_train, y_3_L_val = train_test_split(X_3_temp, y_3_L_temp, test_size=0.25, shuffle=True, random_state=42)

y_3_G_train, y_3_G_val, y_3_G_test = np.transpose(y_3_G_train), np.transpose(y_3_G_val), np.transpose(y_3_G_test)
y_3_L_train, y_3_L_val, y_3_L_test = np.transpose(y_3_L_train), np.transpose(y_3_L_val), np.transpose(y_3_L_test)

#%% Perform upsampling 
def upsample(X_train, y_train):
    X_ones = X_train[y_train.sum(axis=0) >= 1]
    y_ones = np.transpose(y_train)[y_train.sum(axis=0) >= 1]
    n_samples = len(X_train) - 2*len(X_ones)
    X_ones_upsampled = resample(X_ones, random_state=42, n_samples=n_samples, replace=True)
    y_ones_upsampled = resample(y_ones, random_state=42, n_samples=n_samples, replace=True)
    
    return np.append(X_train, X_ones_upsampled, axis=0), np.append(y_train, np.transpose(y_ones_upsampled), axis=1)

X_1_train, y_1_L_train = upsample(X_1_train, y_1_L_train)

#%% Step 4: Classification

def classifiers(X_train, y_train, X_test, y_test, clfs=[svm.SVC()], target="G"):
    accuracies = [{} for clf in clfs]
    classifiers = [{} for clf in clfs]
    predictions = [{} for clf in clfs]
    
    for g in range(len(y_train)):
        for i in range(len(clfs)):
            clf = clfs[i]
            if np.sum(y_train[g]) == 0:
                y_pred = np.zeros(len(X_test))
                print("Boink", g)
            else:
                clf.fit(X_train, y_train[g])
                y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test[g],y_pred)
            accuracies[i][str(target + ":" + str(g))] = accuracy
            classifiers[i][str(target + ":" + str(g))] = clf
            predictions[i][str(target + ":" + str(g))] = y_pred
    
    return accuracies, classifiers, predictions

#%% Tuning regularization hyperparameter for the two models 
def tuneRegu(model_type="linear", C_list=np.linspace(0.1,1,10)):
    accs = []
    for C in C_list:
        if model_type == "linear":
            clfs = [svm.LinearSVC(dual="auto", C=C)]
        elif model_type == "non-linear":
            clfs = [svm.SVC(C=C)]
        acc,_,_ = classifiers(X_1_train, y_1_G_train, X_1_val, y_1_G_val, clfs, target="G")
        accs.append(np.mean(list(acc[0].values()))) 
    
    best_ix = np.argmax(accs)
    best_C = C_list[best_ix]
    
    return accs, best_C

accs_lin, C_lin = tuneRegu(model_type="linear", C_list=np.linspace(1,10,10))
accs_non, C_non = tuneRegu(model_type="non-linear", C_list=np.linspace(10,100,10))

#%% Comparing performance of classifiers 
# Classifiers to compare 
clfs = [svm.LinearSVC(dual="auto"), svm.SVC(), RandomForestClassifier()]
# clfs = [svm.NuSVC(gamma="auto"), RandomForestClassifier()]

### Predict generator status ###
acc_1_G, clf_1_G, y_1_pred_G = classifiers(X_1_train, y_1_G_train, X_1_val, y_1_G_val, clfs, target="G")

### Step 4 - Predict active constraints ###
acc_1_L, clf_1_L, y_1_pred_L = classifiers(X_1_train, y_1_L_train, X_1_val, y_1_L_val, clfs, target="L")

### Predict generator status ###
acc_3_G, clf_3_G, y_3_pred_G = classifiers(X_3_train, y_3_G_train, X_3_val, y_3_G_val, clfs, target="G")

### Step 4 - Predict active constraints ###
acc_3_L, clf_3_L, y_3_pred_L = classifiers(X_3_train, y_3_L_train, X_3_val, y_3_L_val, clfs, target="L")

""" Task: Other classifiers? SVM always predicts 0 currently,
    so something definitely needs to change - or is that the conclusion for
    our linear solver? """

#%% Step 5 - evaluation
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
plot.cf_matrix(y_1_G_val.flatten(), np.asarray(list(y_1_pred_G[0].values())).flatten(), title="Confusion Matrix (generator, 1 hour, Lin SVM)")
plot.cf_matrix(y_1_G_val.flatten(), np.asarray(list(y_1_pred_G[1].values())).flatten(), title="Confusion Matrix (generator, 1 hour, Non-lin SVM)")
plot.cf_matrix(y_1_G_val.flatten(), np.asarray(list(y_1_pred_G[2].values())).flatten(), title="Confusion Matrix (generator, 1 hour, RF)")

plot.cf_matrix(y_1_L_val.flatten(), np.asarray(list(y_1_pred_L[0].values())).flatten(), title="Confusion Matrix (line, 1 hour, Lin SVM)")
plot.cf_matrix(y_1_L_val.flatten(), np.asarray(list(y_1_pred_L[1].values())).flatten(), title="Confusion Matrix (line, 1 hour, Non-lin SVM)")
plot.cf_matrix(y_1_L_val.flatten(), np.asarray(list(y_1_pred_L[2].values())).flatten(), title="Confusion Matrix (line, 1 hour, RF)")

plot.cf_matrix(y_3_G_val.flatten(), np.asarray(list(y_3_pred_G[0].values())).flatten(), title="Confusion Matrix (generator, 3 hour, Lin SVM)")
plot.cf_matrix(y_3_G_val.flatten(), np.asarray(list(y_3_pred_G[1].values())).flatten(), title="Confusion Matrix (generator, 3 hour, Non-lin SVM)")
plot.cf_matrix(y_3_G_val.flatten(), np.asarray(list(y_3_pred_G[2].values())).flatten(), title="Confusion Matrix (generator, 3 hour, RF)")

plot.cf_matrix(y_3_L_val.flatten(), np.asarray(list(y_3_pred_L[0].values())).flatten(), title="Confusion Matrix (line, 3 hour, Lin SVM)")
plot.cf_matrix(y_3_L_val.flatten(), np.asarray(list(y_3_pred_L[1].values())).flatten(), title="Confusion Matrix (line, 3 hour, Non-lin SVM)")
plot.cf_matrix(y_3_L_val.flatten(), np.asarray(list(y_3_pred_L[2].values())).flatten(), title="Confusion Matrix (line, 3 hour, RF)")

#%% Step 6
n_test_days = int(n_samples*0.2)
test_day = 0
test_day = max(min([test_day, n_test_days-1]),0)

demand_test = X_test[24*test_day:24*(test_day+1),91:2*91]

_, _, _, opt_real, t_no_init = uc.uc(demand_test, log=False)
print("Time to solve without any initialization:    ", t_no_init)

""" Task: Evaluation of runtime of unit commitment problem when initialing
    with predicted 'b'-variables. """
pred_G_test = np.asarray([values[24*test_day:24*(test_day+1)] for values in y_pred_G_rf.values()])

_, _, _, opt_g, t_with_init = uc.uc(demand_test, b_pred = pred_G_test)
print("Time to solve with generator initialization: ", t_with_init)

""" Task: Evaluation of runtime of unit commitment problem when trimming
    problem by removing predicted inactive line constraints. """
pred_L_test = np.asarray([values[24*test_day:24*(test_day+1)] for values in y_pred_L_rf.values()]).T

_, _, _, opt_l, t_trim = uc.uc(demand_test, active_lines = pred_L_test)
print("Time to solve with constraint trimming:      ", t_trim)

# All predictions utilised:
_, _, _, opt_all, t_all = uc.uc(demand_test, b_pred = pred_G_test, active_lines = pred_L_test)
print("Time to solve with both model alterations:   ", t_all)

print()
print("Actual optimum:                        ", opt_real)
print("Optimum with trimmed constraints:      ", opt_l)
if (opt_real-opt_l)/opt_real > 0.0001:
    print("Not same solution")

#%% Step 7 - plotting of PCA


plot.pca_plot(X_train, X_test, y_pred_G_rf, gen='G:37', pc = [1,2])

plot.pca_plot(X_train, X_test, y_pred_G_svm, gen='G:37', pc = [1,2])







