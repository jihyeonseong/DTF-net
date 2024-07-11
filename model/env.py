import os
import numpy as np 
import pandas as pd
import gym 
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    

class Env(gym.Env):
    def __init__(self, data, valid_data, args, result_folder):
        self.args = args
        self.MAX_SEQ_LEN = args.max_seq
        self.ratio = args.ratio
        self.regressor = self.args.regressor
        if args.data == 'stock':
            self.data = data.values
            self.columns = data.columns
            if valid_data is not None:
                self.data = np.concatenate((self.data, valid_data.values), axis=0)
        else:
            self.data = data.data.values 
            self.columns = data.data.columns
            if valid_data != None:
                self.data = np.concatenate((self.data, valid_data.data.values), axis=0)
            
        self.segment_length = self.args.seq_len + self.args.pred_len
        self.feature = self.data.shape[1]
        self.state_type = self.args.state
        
        self.action_space = gym.spaces.Discrete(2) 
        self.set_gym_spaces()
        
        self.embedding = PositionalEmbedding(self.args.state_dim)
        
        self.chunk = 0
        self.flag = 0
        self.flag_old = 0
        
        self.reward_list = []
        self.episode_num = 0
        
    def set_gym_spaces(self):
        if self.state_type == 'position':
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.feature, self.args.state_dim))
        else:
            raise ValueError()
        return        
        
    def reset(self, predict=None):
        if predict == True:
            random_episode_index = 0
            seq_len = len(self.data)
        else:
            # select sequence and intialize timestamp as zero
            random_episode_index = np.random.randint(len(self.data)-self.MAX_SEQ_LEN)
            seq_len = np.random.randint(self.segment_length+self.args.label_len, self.MAX_SEQ_LEN)

        self.episode_data = self.data[random_episode_index: random_episode_index+seq_len]

        self.state_trend = [self.data[random_episode_index]] 
        self.feature_trend = [self.data[random_episode_index]]
        self.action_trend = [1] 
        self.timestamp = 1
        
        self.random_reward = np.random.choice(2, seq_len, p=[1-self.ratio, self.ratio])
        #self.random_reward_index = np.arange(0,seq_len, int(seq_len*self.ratio))
        #self.random_reward = list(np.zeros(seq_len))
        #for i in self.random_reward_index:
        #    self.random_reward[i] = 1
        #self.random_reward = np.array(self.random_reward)
        
        self.bi_flag = False
        self.old_action = []
        
        #np.save(os.path.join('./reward', f'reward_{self.episode_num}.npy'), np.array(self.reward_list))
        self.reward_list = []
        self.episode_num += 1
        
        return self.compute_state(None)
        
    def step(self, action):
        point_state = self.episode_data[self.timestamp]
        self.action_trend.append(action)
        self.state_trend.append(point_state)
        if self.bi_flag == False:
            if action == 1:
                self.feature_trend.append(point_state)
                self.chunk = self.timestamp
                self.flag_old = self.flag
                self.flag = self.timestamp
            else:
                self.feature_trend.append([np.nan for i in range(self.feature)])
        else:
            if action == 1 and self.old_action[self.timestamp] == 1:
                self.feature_trend.append(point_state)
                self.chunk = self.timestamp
                self.flag_old = self.flag
                self.flag = self.timestamp
            else:
                self.action_trend[-1] = 0
                self.feature_trend.append([np.nan for i in range(self.feature)])
        
        feature_trend = self.feature_trend
        action_trend = None #self.action_trend
            
        if True == self.random_reward[self.timestamp]:
            reward = self.compute_reward(feature_trend)
            self.reward_list.append(reward)
        else:
            reward = 0
        
        state = self.compute_state(action_trend)

        done = self.compute_done()
        info = {} 
        
        self.timestamp +=1 
        return state, reward, done, info 
    
    def compute_state(self, action_trend):
        if self.state_type == 'position':
            state_trend = np.array(self.state_trend)
            if action_trend == None:
                action_trend = np.repeat(np.array(self.action_trend).reshape(-1,1), self.feature, axis=1)
            emb = np.concatenate((state_trend, action_trend), axis=0)
            state = self.embedding(torch.Tensor(emb))
        else:
            raise ValueError()
        return state
        
    def compute_reward(self, feature_trend):
        def data_processor():
            feature_segment = pd.DataFrame(feature_trend).interpolate(direction='both', method='linear').fillna(method="ffill").fillna(method="bfill").values
            X = np.concatenate((np.array(self.state_trend), feature_segment), axis=1)
            y = np.array(self.episode_data)
            return X, y
        
        def indexing():
            token = 0
            if self.timestamp < self.args.seq_len+self.args.label_len: 
                self.chunk = 0
            else:
                tmp = self.chunk
                if self.timestamp - tmp < self.args.seq_len+self.args.label_len: 
                    self.chunk = tmp - ((self.args.seq_len+self.args.label_len) - (self.timestamp - tmp)) 
                elif self.timestamp - tmp > self.args.seq_len+self.args.label_len: 
                    self.chunk = tmp + ((self.timestamp-tmp) - (self.args.seq_len+self.args.label_len)) 
                if self.timestamp - self.chunk == 0:
                    self.chunk = self.timestamp - self.args.seq_len - self.args.label_len
            if self.flag <= self.chunk:
                self.chunk = self.flag_old
            if self.chunk > self.args.label_len:
                token = self.args.label_len
            return token
        
        X, y = data_processor()
        token = indexing()
        X = X[self.chunk-token:self.timestamp]
        y = y[self.chunk+self.args.pred_len-token: self.timestamp + self.args.pred_len]
        
        reward=0
        if self.regressor == 'Linear':
            model = LinearRegression()
            model.fit(X, y)
            pred = model.predict(X)
            loss = np.square(np.subtract(y, pred)).mean()
            reward = -loss
        elif self.regressor == 'SVM':
            model = SVR(C=1.0, epsilon=0.2)
            model.fit(X,y)
            pred = model.predict(X)
            loss = np.square(np.subtract(y, pred)).mean()
            reward = -loss
        elif self.regressor == 'ARIMA' and self.timestamp > self.args.seq_len:
            idx = np.arange(len(X))
            X = pd.DataFrame(X, index=idx).iloc[:, 1]
            model = ARIMA(X, order=(0,1,1))
            model_fit = model.fit()
            pred = model_fit.forecast(len(y))
            loss = np.square(np.subtract(y, pred.values)).mean()
            reward = -loss            
        else:
            reward = 0 
        return reward 
    
    def compute_done(self):
        if self.timestamp == len(self.episode_data)-1-self.args.pred_len and self.args.bidirection == True and self.bi_flag == False:
            self.timestamp=1
            self.old_action = self.action_trend
            self.state_trend = [self.state_trend[0]] 
            self.feature_trend = [self.feature_trend[0]]
            self.action_trend = [1]
            self.bi_flag=True
            done=False
        elif self.timestamp == len(self.episode_data)-1-self.args.pred_len:
            done = True
        else:
            done = False
        return done 
    
    def get_data(self):
        return pd.DataFrame(np.array(self.episode_data)[:, :self.timestamp], columns=self.columns), pd.DataFrame(self.feature_trend, columns=self.columns), pd.DataFrame(np.array(self.action_trend), columns=['point'])
