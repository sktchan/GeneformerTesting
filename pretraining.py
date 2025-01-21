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

# cpu cores
num_proc = 16

# training dataset size
subsample_size = 10_000

# load gene_ensembl_id:token dictionary
with open("genecorpus_30M/token_dictionary.pkl", "rb") as fp:
    token_dictionary = pickle.load(fp)
token_gene_dict = {v: k for k,v in token_dictionary.items()}

# prepare targets and labels
def prep_inputs(genegroup1, genegroup2, balance):

    targets1 = [gene for gene in genegroup1 if gene in token_dictionary]
    targets2 = [gene for gene in genegroup2 if gene in token_dictionary]
    
    if balance == "balance":
        min_sample = min(len(targets1), len(targets2))
        random.seed()
        targets1 = random.sample(targets1, min_sample)
        random.seed()
        targets2 = random.sample(targets2, min_sample)

    targets1_id = [token_dictionary[gene] for gene in targets1]
    targets2_id = [token_dictionary[gene] for gene in targets2]
    
    targets = np.array(targets1_id + targets2_id)
    labels = np.array([0]*len(targets1_id) + [1]*len(targets2_id))
    nsplits = min(5, min(len(targets1_id), len(targets2_id))-1)
    assert nsplits > 2
    print(f"# targets1: {len(targets1_id)}\n# targets2: {len(targets2_id)}\n# splits: {nsplits}")
    return targets, labels, nsplits

### CHANGED CODE ###
# dosage_sens_tfs is a pickle, not csv. 
# also needed to convert lists to pandas Series objects to dropna().
dosage_tfs_pickle = pd.read_pickle("genecorpus_30M/example_input_files/gene_classification/dosage_sensitive_tfs/dosage_sensitivity_TFs.pickle")
sensitive = pd.Series(dosage_tfs_pickle["Dosage-sensitive TFs"]).dropna()
insensitive = pd.Series(dosage_tfs_pickle["Dosage-insensitive TFs"]).dropna()
targets, labels, nsplits = prep_inputs(sensitive, insensitive, "balance")

# load training dataset
train_dataset=load_from_disk("genecorpus_30M/genecorpus_30_2048.dataset")
shuffled_train_dataset = train_dataset.shuffle()

# reduce training dataset to 5x subsample size (to leave room for further filtering for cells that express target genes)
subsampled_training_dataset = shuffled_train_dataset.select([i for i in range(subsample_size*5)])

def if_contains_target(example):
    a = targets
    b = example['input_ids']
    return not set(a).isdisjoint(b)

# filter dataset for cells that express target genes
data_w_target = subsampled_training_dataset.filter(if_contains_target, num_proc=num_proc)

# subsample data to desired number of training examples
data_subsample = data_w_target.select([i for i in range(subsample_size)])

def get_ranks(example):
    example_rank_dict = dict(zip(example["input_ids"],[2048-i for i in range(example["length"])]))
    target_vector = [example_rank_dict.get(target,0) for target in targets]
    example["target_vector"] = target_vector
    return example

# get ranks of target genes within training cells for rank-based trials
data_w_target_vectors = data_subsample.map(get_ranks, num_proc=num_proc)
target_arr = np.transpose(np.array(data_w_target_vectors["target_vector"]))

np.save("target_arr.npy", target_arr)
np.save("labels.npy", labels)