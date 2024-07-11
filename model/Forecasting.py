import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import sys 
sys.path.append("../")
import os
from utils.FeatureEngineering import FeatureEngineering
from model.DLinear import DLinear, NLinear

device = 'cuda'

SEED = 10
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

class MyDataset(Dataset):
    def __init__(self, args, X, y, window, offset, token):
        self.args = args
        self.target_index = X.columns.get_loc(args.target)
        
        self.data1 = torch.Tensor(X.values)
        self.answer = torch.Tensor(y)
        self.window = window 
        self.token = token
        
        self.offset = offset
        self.shape = self.__getshape__()
        self.size = self.__getsize__()

    def __getitem__(self, index):
        x1 = self.data1[index:index+self.window, :]
        y = self.answer[index+self.window-self.token: index+self.window+self.offset] 
        
        return x1, y

    def __len__(self):
        return len(self.answer) -  self.window - self.offset # train data length
    
    def __getshape__(self):
        return (self.__len__(), *self.__getitem__(0)[0].shape) # row, col
    
    def __getsize__(self):
        return (self.__len__())
    
    
def forecasting(args, X, y, features_n, out_n, predict = False, result_folder=None):       
    if args.model == 'DLinear':
        model = DLinear(in_features = features_n,
                    out_features=out_n, args=args).cuda()
    elif args.model == 'NLinear':
        model = NLinear(in_features = features_n,
                    out_features=out_n, args=args).cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    if predict == True:
        train_dataset = MyDataset(args, X, y.values, 
                                  args.seq_len, args.pred_len, args.label_len)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                             drop_last=False, 
                                             num_workers=4, pin_memory=True)

        checkpoint = torch.load(os.path.join(result_folder, f'prediction-Nlinear-best.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['loss']

        answer = []
        prediction = []
        loss_list = []
        model = model.eval()
        with torch.no_grad():
            for i, (x1, y) in enumerate(train_loader):
                x1 = x1.to(device)
                y = y.to(device)
                
                outputs = model(x1)
                loss = criterion(outputs.squeeze(), y.squeeze())
                
                answer.extend(y[:, args.label_len:].squeeze().detach().cpu().numpy())
                prediction.extend(outputs[:, args.label_len:].squeeze().detach().cpu().numpy())
                loss_list.append(loss.item())

        answer2 = torch.Tensor(answer)
        prediction2 = torch.Tensor(prediction)

        loss = criterion(answer2, prediction2)
        criterion3 = nn.L1Loss()
        loss3 = criterion3(input=prediction2, target = answer2)

        loss4 = torch.mean(torch.abs(torch.subtract(prediction2, answer2) / answer2))
        final_loss = loss.item()
        
        pd.DataFrame([loss.item(), torch.sqrt(loss).item(), loss3.item(), loss4.item()], 
                     index=['MSE', 'rMSE', 'MAE', 'MAPE']).to_csv(os.path.join(result_folder, 'perf.csv'))
        
        pd.DataFrame(vars(args), index=np.arange(18)).to_csv(os.path.join(result_folder, 'log.csv'))

        print("MSE: ", loss.item())
        print("rMSE: ", torch.sqrt(loss).item())
        print("MAE: ", loss3.item())
        print("MAPE: ", loss4.item())


    else:
        train_dataset = MyDataset(args, 
                                  X[0], y[0].values, 
                                  args.seq_len, args.pred_len, args.label_len)
        valid_dataset = MyDataset(args,
                                  X[1], y[1].values, 
                                  args.seq_len, args.pred_len, args.label_len)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                             drop_last=False, 
                                             num_workers=4, pin_memory=True, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, 
                                             drop_last=False, 
                                             num_workers=4, pin_memory=True)
        answer = []
        prediction = []
        loss_list = []
        
        best_loss = 10 ** 9 
        patience_limit = 3 
        patience_check = 0 
        for epoch in tqdm(range(0, args.net_epoch)):
            train_loss = []
            answer = []
            prediction = []
            model = model.train()
            for i, (x1, y) in enumerate(train_loader):
                x1 = x1.cuda()
                y = y.cuda()
                
                outputs = model(x1)
                loss = criterion(outputs.squeeze(), y.squeeze())
                
                answer.extend(y[:, args.label_len:].squeeze().detach().cpu().numpy())
                prediction.extend(outputs[:, args.label_len:].squeeze().detach().cpu().numpy())

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_loss.append(loss.item())
                
            valid_loss = []
            model = model.eval()
            with torch.no_grad():
                for i, (x1, y) in enumerate(valid_loader):
                    x1 = x1.cuda()
                    y = y.cuda()
                    
                    outputs = model(x1)
                    loss = criterion(outputs.squeeze(), y.squeeze())
                    
                    answer.extend(y[:, args.label_len:].squeeze().detach().cpu().numpy())
                    prediction.extend(outputs[:, args.label_len:].squeeze().detach().cpu().numpy())
                    
                    valid_loss.append(loss.item())

            loss_list.append(np.mean(valid_loss))
            if (epoch == 0) or (epoch>0 and (min(loss_list[:-1]) > loss_list[-1])):
                torch.save({
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optim.state_dict(),
                'loss' : criterion
                }, os.path.join(result_folder, f'prediction-Nlinear-best.pt'))
            """    
            ### early stopping ###
            if np.mean(valid_loss) > best_loss: 
                patience_check += 1

                if patience_check >= patience_limit: 
                    break

            else: 
                best_loss = np.mean(valid_loss)
                patience_check = 0
            #"""
        answer2 = torch.Tensor(answer)
        prediction2 = torch.Tensor(prediction)

        loss = criterion(answer2, prediction2)
        criterion3 = nn.L1Loss()
        loss3 = criterion3(input=prediction2, target = answer2)

        final_loss = loss.item() 

        print("MAE Loss: ", loss3.item())

    return final_loss, answer, prediction