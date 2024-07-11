import numpy as np
import pandas as pd
from matplotlib import figure
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import time
import random
import gc
import statistics
import json
import argparse
from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.data_factory import data_provider
from model.env import Env
from stable_baselines3 import PPO, DQN, A2C

from baselines.Forecasting import forecasting

import datetime
import sys
sys.setrecursionlimit(15000)

random_seed=2023
torch.manual_seed(random_seed) 
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 
np.random.seed(random_seed) 
random.seed(random_seed) 
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) 

device = 'cuda'
data_folder = '/data/RLTF/weather.csv/'
result_folder = f'/data/RLTF/weather/{datetime.datetime.now()}'
os.makedirs(result_folder, exist_ok=True)  

def main(args):            
    origin = pd.read_csv(os.path.join(data_folder, 'rl_result_train.csv'), index_col=0)            
    origin_vali = pd.read_csv(os.path.join(data_folder, 'rl_result_valid.csv'), index_col=0) 
    
    X = []
    y = []
    dim = 0
    out = 0
    if args.forecasting == 'M':
        dim = train_data.data.shape[1]*2
        out = dim//2
        y = [origin, origin_vali]
        X = [origin, origin_vali]
    elif args.forecasting == 'MS':
        out = 1
        dim = train_data.data.shape[1]+1
        y = [origin.iloc[:, -1], origin_vali.iloc[:, -1]]
        X = [origin, origin_vali]
    else:
        dim = 2
        out = 1
        y = [origin.iloc[:, [-1]], origin_vali.iloc[:, [-1]]]
        X = [origin.iloc[:, [-1]], origin_vali.iloc[:, [-1]]]
    
    loss, answer, prediction = forecasting(args, X, y, dim, out, False, result_folder)
    
    np.save(os.path.join(result_folder, 'answer_train'), answer)
    np.save(os.path.join(result_folder, 'pred_train'), prediction)
    
    
    print("Evaluation with test data...")    
    origin = pd.read_csv(os.path.join(data_folder, 'rl_result_test.csv'), index_col=0) 

    X = []
    y = []
    if args.forecasting == 'M':
        y = origin
        X = pd.concat([origin, trend], axis=1)
    elif args.forecasting == 'MS':
        y = origin.iloc[:, -1]
        X = origin 
    else:
        y = origin.iloc[:, [-1]]
        X = origin.iloc[:, [-1]] 
    
    loss, answer, prediction = forecasting(args, X, y, dim, out, True, result_folder)
    
    np.save(os.path.join(result_folder, 'answer_test'), answer)
    np.save(os.path.join(result_folder, 'pred_test'), prediction)

        
        

if __name__=='__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuidx', default=1, type=int, help='gpu index')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    
    parser.add_argument('--data', default='ETTh1', type=str, help='data name')
    parser.add_argument('--embed', default='timeF', type=str, help='embedding')
    parser.add_argument('--freq', default='h', type=str, help='frequency')
    parser.add_argument('--root_path', default=data_folder, type=str, help='data folder')
    parser.add_argument('--data_path', default='ETTh1.csv', type=str, help='data file')
    parser.add_argument('--seq_len', default=336, type=int, help='window')
    parser.add_argument('--label_len', default=0, type=int, help='for decoder')
    parser.add_argument('--pred_len', default=24, type=int, help='pred len')
    parser.add_argument('--features', default='S', type=str, help='for RL uni | multi')
    parser.add_argument('--target', default='OT', type=str, help='target')
    
    parser.add_argument('--forecasting', default='S', type=str, help='uni | multi')
    parser.add_argument('--model', default='NLinear', type=str, help='forecasting model')
    
    parser.add_argument('--net_epoch', default=15, type=int, help='forecasting training')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='learning rate')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
    
    main(args)