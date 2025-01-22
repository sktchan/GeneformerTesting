# imports
import os
import numpy as np
import pandas as pd
import random
import datetime
import subprocess
import math
import pickle
from tqdm.notebook import tqdm
import anndata
import scanpy as sc
from datasets import load_from_disk

# visualization
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix

# data
from config import target_arr, labels, nsplits, subsample_size
target_arr = np.array(target_arr)
labels = np.array(labels)
labels_pred = pd.read_csv("labels_pred.csv")
labels_eval = pd.read_csv("labels_eval.csv")

# calculating metrics for new model
fpr, tpr, thresholds = roc_curve(labels_eval, labels_pred)
roc_auc = auc(fpr, tpr)
roc_auc_sd = 0.0 # since they calculated it


# geneformer benchmarking models to compare against

# functions to evaluate classifier
def classifier_predict(model_type, model, eval_arr, labels_eval, mean_fpr):
    y_pred = model.predict(eval_arr)
    y_true = labels_eval
    conf_mat = confusion_matrix(y_true, y_pred)
    # probability of class 1
    if model_type == "SVM":
        y_prob = model.decision_function(eval_arr)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    else:
        y_prob = model.predict_proba(eval_arr)
        fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
    # plot roc_curve for this split
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()
    # interpolate to graph
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    return fpr, tpr, interp_tpr, conf_mat 

# get cross-validated AUC mean and sd metrics
def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
    wts = [count/sum(all_tpr_wt) for count in all_tpr_wt]
    print(wts)
    all_weighted_tpr = [a*b for a,b in zip(all_tpr, wts)]
    mean_tpr = np.sum(all_weighted_tpr, axis=0)
    mean_tpr[-1] = 1.0
    all_weighted_roc_auc = [a*b for a,b in zip(all_roc_auc, wts)]
    roc_auc = np.sum(all_weighted_roc_auc)
    roc_auc_sd = math.sqrt(np.average((all_roc_auc-roc_auc)**2, weights=wts))
    return mean_tpr, roc_auc, roc_auc_sd

# cross-validate token classifier
def cross_validate(model_type, model, target_arr, labels, nsplits, subsample_size, num_proc):
    print(f"# training cells: {target_arr.shape[1]}")
    
    # initiate eval metrics to return
    num_classes = len(set(labels))
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_roc_auc = []
    all_tpr_wt = []
    confusion = np.zeros((num_classes,num_classes))
    
    # set up cross-validation splits
    skf = StratifiedKFold(n_splits=nsplits, shuffle=True)
    # train and evaluate
    iteration_num = 0
    for train_index, eval_index in tqdm(skf.split(target_arr, labels)):

        print(f"****** Crossval split: {iteration_num}/{nsplits-1} ******\n")
        # generate cross-validation splits
        targets_train, targets_eval = target_arr[train_index], target_arr[eval_index]
        labels_train, labels_eval = labels[train_index], labels[eval_index]
        
        model = model
    
        # train the token classifier
        model.fit(targets_train, labels_train)

        # evaluate model
        fpr, tpr, interp_tpr, conf_mat = classifier_predict(model_type, model, targets_eval, labels_eval, mean_fpr)

        # append to tpr and roc lists
        confusion = confusion + conf_mat
        all_tpr.append(interp_tpr)
        all_roc_auc.append(auc(fpr, tpr))
        # append number of eval examples by which to weight tpr in averaged graphs
        all_tpr_wt.append(len(tpr))
        
        iteration_num = iteration_num + 1
        
    # get overall metrics for cross-validation
    mean_tpr, roc_auc, roc_auc_sd = get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt)
    return all_roc_auc, roc_auc, roc_auc_sd, mean_fpr, mean_tpr, confusion

# plot ROC curve
def plot_ROC(bundled_data, title):
    fig = plt.figure()
    fig.set_size_inches(17, 10.5)
    plt.rcParams.update({'font.size': 20})
    lw = 4
    for roc_auc, roc_auc_sd, mean_fpr, mean_tpr, sample, color, linestyle in bundled_data:
        plt.plot(mean_fpr, mean_tpr, color=color,
                 lw=lw, label="{0} (AUC {1:0.2f} $\pm$ {2:0.2f})".format(sample, roc_auc, roc_auc_sd), linestyle=linestyle)
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")

    plt.show()

model = RandomForestClassifier(max_depth=2)
all_roc_auc0, roc_auc0, roc_auc_sd0, mean_fpr0, mean_tpr0, confusion0 \
    = cross_validate("RF", model, target_arr, labels, nsplits, subsample_size, 1)

model = LogisticRegression()
all_roc_auc1, roc_auc1, roc_auc_sd1, mean_fpr1, mean_tpr1, confusion1 \
    = cross_validate("LR", model, target_arr, labels, nsplits, subsample_size, 1)

model = SVC()
all_roc_auc2, roc_auc2, roc_auc_sd2, mean_fpr2, mean_tpr2, confusion2 \
    = cross_validate("SVM", model, target_arr, labels, nsplits, subsample_size, 1)

# bundle data for plotting
bundled_data = []
bundled_data += [(roc_auc2, roc_auc_sd2, mean_fpr2, mean_tpr2, "SVM rank", "purple", "solid")]
bundled_data += [(roc_auc0, roc_auc_sd0, mean_fpr0, mean_tpr0, "Random Forest rank", "blue", "solid")]
bundled_data += [(roc_auc1, roc_auc_sd1, mean_fpr1, mean_tpr1, "Logistic Regression rank", "green", "solid")]
bundled_data += [(roc_auc, roc_auc_sd, fpr, tpr, "Simple FNN", "orange", "solid")]

# plot ROC
plot_ROC(bundled_data, 'Dosage Sensitive vs Insensitive TFs')
plt.savefig('roc_plot.png')