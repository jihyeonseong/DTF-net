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

from model.Forecasting import forecasting

import datetime
import sys
sys.setrecursionlimit(15000)

device = 'cuda'

data_folder = '/data/RLTF/data/exchange_rate/'
result_folder = f'/data/RLTF_final/ETTh1/{datetime.datetime.now()}'
#os.makedirs(result_folder, exist_ok=True)  

def main(args):
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    epoch = args.epoch
    env = Env(train_data, vali_data, args, result_folder)
    RLmodel = PPO("MlpPolicy", env, verbose=1, device='cuda')
    print("RL training...")
    RLmodel.learn(total_timesteps=args.epoch, progress_bar=True) 
    RLmodel.save(os.path.join(result_folder, f"RL_best"))
    
    print("Training Network...")
    env1 = Env(train_data, None, args, result_folder)
    obs = env1.reset(predict=True).reshape(train_data.data.shape[1], args.state_dim)
    model = PPO.load(os.path.join(result_folder, f"RL_best"))
    model.set_env(env1)
    while range(1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env1.step(action)
        obs = obs.reshape(train_data.data.shape[1], args.state_dim)
        if done:
            break
            
    origin, trend, action = env1.get_data()
    trend = trend.interpolate(direction='both', method='linear')
    trend = trend.fillna(method='ffill') 
    trend = trend.fillna(method='bfill') 

    origin.to_csv(os.path.join(result_folder, 'rl_result_train.csv'))
    trend.to_csv(os.path.join(result_folder, 'trend_train.csv'))
    action.to_csv(os.path.join(result_folder, 'action_point.csv'))

    env2 = Env(vali_data, None, args, result_folder)
    obs = env2.reset(predict=True).reshape(train_data.data.shape[1], args.state_dim)
    model = PPO.load(os.path.join(result_folder, f"RL_best"))
    model.set_env(env2)
    while range(1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env2.step(action)
        obs = obs.reshape(train_data.data.shape[1], args.state_dim)
        if done:
            break
            
    origin_vali, trend_vali, action_valid = env2.get_data()
    trend_vali = trend_vali.interpolate(direction='both', method='linear')
    trend_vali = trend_vali.fillna(method='ffill') 
    trend_vali = trend_vali.fillna(method='bfill') 

    origin_vali.to_csv(os.path.join(result_folder, 'rl_result_valid.csv'))
    trend_vali.to_csv(os.path.join(result_folder, 'trend_valid.csv'))
    action_valid.to_csv(os.path.join(result_folder, 'action_point_valid.csv'))
    
    X = []
    y = []
    dim = 0
    out = 0
    if args.forecasting == 'M':
        dim = train_data.data.shape[1]*2
        out = dim//2
        y = [origin, origin_vali]
        X = [pd.concat([origin, trend], axis=1), 
             pd.concat([origin_vali, trend_vali], axis=1)]
    elif args.forecasting == 'MS':
        out = 1
        dim = train_data.data.shape[1]+1
        y = [origin.iloc[:, -1], origin_vali.iloc[:, -1]]
        X = [pd.concat([origin, trend.iloc[:, -1]], axis=1),
             pd.concat([origin_vali, trend_vali.iloc[:, -1]], axis=1)]
    else:
        dim = 2
        out = 1
        y = [origin.iloc[:, -1], origin_vali.iloc[:, -1]]
        X = [pd.concat([origin.iloc[:, -1], trend.iloc[:, -1]], axis=1), 
             pd.concat([origin_vali.iloc[:, -1], trend_vali.iloc[:, -1]], axis=1)]
    
    loss, answer, prediction = forecasting(args, X, y, dim, out, False, result_folder)
    
    np.save(os.path.join(result_folder, 'answer_train'), answer)
    np.save(os.path.join(result_folder, 'pred_train'), prediction)
    
    
    print("Evaluation with test data...")
    env3 = Env(test_data, None, args, result_folder)
    obs = env3.reset(predict=True).reshape(train_data.data.shape[1], args.state_dim)
    model = PPO.load(os.path.join(result_folder, f"RL_best"))
    model.set_env(env3)
    while range(1):
        action, _states = model.predict(obs)
        obs, reward, done, info = env3.step(action)
        obs = obs.reshape(train_data.data.shape[1], args.state_dim)
        if done: 
            break
    
    origin, trend, action = env3.get_data()
    """
    ################ [start] long-heavy tail min/max remove [test] ################
    trend.to_csv(os.path.join(result_folder, 'trend_points_test.csv')) 
    trend_points = trend.copy()
    remove_percent = 0.15 # 0: non-remove, 0.1: min max each 10% remove(total 20%)
    minmax_count = int(trend_points.iloc[:,-1].count() * remove_percent) # trend_points = original_trend_points
    trend_points_maxgroup = trend_points.nlargest(minmax_count, args.target, keep = "all") # OT: self.target
    trend_points_max_group_index = trend_points_maxgroup.index  #max 10% index
    trend_points_mingroup = trend_points.nsmallest(minmax_count, args.target, keep = "all") # OT: self.target
    trend_points_min_group_index = trend_points_mingroup.index #min 10% index
    max_group_nan_list = [np.NaN for i in range(len(trend_points_max_group_index))]
    trend_points_without_max = trend_points.replace(trend_points.iloc[trend_points_max_group_index].OT.values,max_group_nan_list) #remove max
    min_group_nan_list = [np.NaN for i in range(len(trend_points_min_group_index))]
    trend_points_without_minmax = trend_points_without_max.replace(trend_points_without_max.iloc[trend_points_min_group_index].OT.values,min_group_nan_list) #remove min
    trend = trend_points_without_minmax.copy()
    trend.to_csv(os.path.join(result_folder, 'trend_points_test_without_minmax.csv'))
    ################ [end] long-heavy tail min/max remove by osk end ################
    #"""
    trend = trend.interpolate(direction='both', method='linear')
    trend = trend.fillna(method='ffill') 
    trend = trend.fillna(method='bfill') 
    print(np.isnan(trend.values).sum())
    origin.to_csv(os.path.join(result_folder, 'rl_result_test.csv'))
    trend.to_csv(os.path.join(result_folder, 'trend_test.csv'))
    action.to_csv(os.path.join(result_folder, 'action_point_test.csv'))
    
    X = []
    y = []
    if args.forecasting == 'M':
        y = origin
        X = pd.concat([origin, trend], axis=1)
    elif args.forecasting == 'MS':
        y = origin.iloc[:, -1]
        X = pd.concat([origin, trend.iloc[:, -1]], axis=1) 
    else:
        y = origin.iloc[:, -1]
        X = pd.concat([origin.iloc[:, -1], trend.iloc[:, -1]], axis=1) 
    
    loss, answer, prediction = forecasting(args, X, y, dim, out, True, result_folder)
    
    np.save(os.path.join(result_folder, 'answer_test'), answer)
    np.save(os.path.join(result_folder, 'pred_test'), prediction)

        
        

if __name__=='__main__':
    # Arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2023, type=int, help='random seed')
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
    parser.add_argument('--regressor', default='Linear', type=str, help='ML forecasting model in ENV')
    parser.add_argument('--state', default='position', type=str, help='state encoding type')
    parser.add_argument('--state_dim', default=300, type=int, help='state dimension')
    parser.add_argument('--max_seq', default=3000, type=int, help='max sequence')
    parser.add_argument('--ratio', default=0.1, type=float, help='reward ratio')
    parser.add_argument('--bidirection', default=False, action='store_true', help='bidirection')
    parser.add_argument('--model', default='NLinear', type=str, help='forecasting model')
    
    
    parser.add_argument('--epoch', default=10000, type=int, help='training num')
    parser.add_argument('--net_epoch', default=15, type=int, help='forecasting training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--interpolation', default='linear', type=str, help='interpolation method')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuidx)
    
    random_seed=args.seed
    torch.manual_seed(random_seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    np.random.seed(random_seed) 
    random.seed(random_seed) 
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    
    result_folder = f'/data/RLTF_final/{args.data_path}/{args.ratio}_{args.pred_len}_{datetime.datetime.now()}'
    data_folder = args.root_path
    os.makedirs(result_folder, exist_ok=True)  
    
    pd.DataFrame(vars(args), index=np.arange(1)).to_csv(os.path.join(result_folder, 'log.csv'))
    
    main(args)