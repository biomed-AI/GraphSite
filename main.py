import os
import pickle
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from time import time
import datetime, random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler

from model import *
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='./Dataset/')
parser.add_argument("--feature_path", type=str, default='./Feature/')
parser.add_argument("--use_apm", action='store_true', default=False)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--train", action='store_true', default=False)
parser.add_argument("--test1", action='store_true', default=False)
parser.add_argument("--test2", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_id", type=str, default=None)

args = parser.parse_args()

seed = args.seed
root = args.feature_path + 'input/'
Dataset_Path = args.dataset_path
Feature_Path = args.feature_path
run_id = args.run_id

Seed_everything(seed=seed)

train_df = pd.read_csv(Dataset_Path + 'train.csv')
test_df1 = pd.read_csv(Dataset_Path + 'test1.csv')
test_df2 = pd.read_csv(Dataset_Path + 'test2.csv')

if args.train:
    ID_list = list(set(train_df['ID']) | set(test_df1['ID']))
elif args.test1:
    ID_list = list(set(test_df1['ID']))
elif args.test2:
    ID_list = list(set(test_df2['ID']))

all_protein_data = {}
for pdb_id in ID_list:
    all_protein_data[pdb_id] = torch.load(root+f"{pdb_id}_node_feature.tensor"), torch.load(root+f"{pdb_id}_edge_feature.tensor"), torch.load(root+f"{pdb_id}_dist.tensor"), torch.load(root+f"{pdb_id}_mask.tensor"), torch.load(root+f"{pdb_id}_label.tensor")

nn_config = {
    'hidden_unit': 64,
    'fc_layer': 2,
    'self_atten_layer': 2,
    'attention_heads': 2,
    'num_neighbor': 30,
    'fc_dropout': 0.2,
    'attention_dropout': 0,
    'node_dim': 384 + 14, # alphafold_node_feature + dssp
    'id_name':'ID', # column name in dataframe
    'obj_max': 1,   # optimization object: max is better
    'epochs': 14,
    'smoothing': 0.0,
    #'clipnorm': 1,
    'patience': 4,
    'lr': 3e-4,
    'batch_size': 16,
    'folds': 5,
    'seed': seed,
    'remark': 'DNA binding site prediction'
}

if args.train:
    oof, sub = NN_train_and_predict(train_df, test_df1, all_protein_data, GTM, nn_config, logit = True, run_id = run_id, args=args)
elif args.test1:
    oof, sub = NN_train_and_predict(None, test_df1, all_protein_data, GTM, nn_config, logit = True, run_id = run_id, args=args)
elif args.test2:
    oof, sub = NN_train_and_predict(None, test_df2, all_protein_data, GTM, nn_config, logit = True, run_id = run_id, args=args)
