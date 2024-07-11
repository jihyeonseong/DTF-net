import numpy as np
import math
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class FeatureEngineering:
    def __init__(self, stock_name):
        data = pd.read_csv(stock_name)
        self.df = data[['close']].copy()
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        #self.breakpoint = breakpoint
        #self.data = pd.DataFrame(self.data[['index', 'close']])

    #simple moving average
    def SMA(self,  data, period=30, column='close'):
        return data[column].rolling(window=period).mean()

    #exponential moving average
    def EMA(self, data, period=20, column='close'):
        return data[column].ewm(span=period, adjust=False).mean()
    
    def Bollingerband(self, data, period=20, column='close'):
        data['SMA'] = self.SMA(data, period=period, column=column)
        data['UB'] = data['SMA'] + 2*data['SMA'].rolling(20).std()
        data['LB'] = data['SMA'] - 2*data['SMA'].rolling(20).std()
        
        return data

    def MACD(self, data, period_long=26, period_short=12, period_signal=9, column='close'):
        data['ShortEMA'] = self.EMA(data, period_short, column=column)

        data['LongEMA'] = self.EMA(data, period_long, column=column)

        data['MACD'] = data['ShortEMA']- data['LongEMA']

          #signal
        data['Signal_Line'] = self.EMA(data, period_signal, column='MACD')

        return data
    
    def Momentum(self, data, period=7, column='close'):
        data['7D'] = data[column].shift(period)
        data['1D'] = data[column].shift(1)
        data['Momentum'] = data['1D'] / data['7D'] - 1
        
        return data
    
    def RSI(self, data, period=14, column='close'):
        delta = data[column].diff(1)
        delta = delta.dropna()

        up = delta.copy()
        down = delta.copy()
        up[up<0] = 0
        down[down>0] = 0
        data['up'] = up
        data['down'] = down

        AVG_Gain = self.SMA(data, period, column='up')
        AVG_Loss = abs(self.SMA(data, period, column='down'))
        RS = AVG_Gain / AVG_Loss

        RSI = 100.0 - (100.0/(1.0+RS))
        data['RSI'] = RSI
  
        return data
    
    #get trading signal
    def trading(self, book): 
        for i in tqdm(book.index):
            if book.loc[i, 'Signal_Line'] > book.loc[i, 'MACD']: #beyond upper band: no action
                book.loc[i, 'signal'] = ''
            elif book.loc[i, 'MACD'] > book.loc[i, 'Signal_Line']: #below lower band: action occur
                if book.shift(1).loc[i, 'signal'] == 'buy': #already holding
                    book.loc[i, 'signal'] = 'buy' #hold
                else: 
                    book.loc[i, 'signal'] = 'buy'
        return book
    
    #actual trading with MACD
    def returns(self, book): 
        rtn = 1.0
        book['return'] = 1
        book['trade'] = ''
        buy = 0.0
        sell = 0.0

        for i in tqdm(book.index):
            if book.loc[i, 'signal'] == 'buy' and book.shift(1).loc[i, 'signal'] == '' and buy == 0.0: #buy signal occur
                #long position
                buy = book.loc[i, 'close']
                book.loc[i, 'trade'] = 'buy'
                
            elif (book.loc[i, 'signal'] == '' and book.shift(1).loc[i, 'signal']=='buy' and buy != 0.0)\
            or (book.loc[i, 'signal'] == 'end' and buy != 0.0): #beyond upper band: no buy signal
                #short position
                sell = book.loc[i, 'close']
                book.loc[i, 'trade'] = 'sell'
                rtn = (sell - buy ) / buy + 1 #return ratio
                #rtn -= rtn * (0.001) #transaction cost
                book.loc[i, 'return'] = rtn
                
                sell = 0.0
                buy = 0.0

        acc_rtn=1.0
        for i in book.index:
            if book.loc[i, 'return']:
                rtn = book.loc[i, 'return']
                acc_rtn = acc_rtn*rtn
            book.loc[i, 'acc return'] = acc_rtn
                
        return book
          
    
    def get_data(self):
        print('Feature Engineering...')
        
        self.train = self.df[:48000]
        self.test = self.df[48000:]
        
        #train
        print("\nTrain dataset")
        self.train['EMA'] = self.EMA(self.train)
        self.train = self.Bollingerband(self.train)
        self.train = self.MACD(self.train)
        self.train = self.Momentum(self.train)
        self.train = self.RSI(self.train)
        
        self.train.dropna(inplace=True)
        self.train.reset_index(inplace=True)
        self.train.drop(columns=['index'], inplace=True)
        
        print("MACD trading...")
        macd = self.train.copy()
        print("Checking signal...", end = ' ')
        if len(macd):
            macd['signal'] = ''
            macd = self.trading(macd)
        if len(macd)!=0:
            macd.loc[0, 'signal'] = ''
            macd.loc[len(macd)-1, 'signal'] = 'end'
        print("Actual trading...", end = ' ')
        macd = self.returns(macd)
        self.train['macd_acc'] = macd['acc return']
        
        #self.train = self.break_point(self.train)
        #move = self.train['break point']
        #self.train.drop(columns = ['break point'], inplace=True)
        #self.train['break point'] = move
        
        #self.train['rl_predict'] = np.nan
        
        #test
        print("\nTest dataset")
        self.test['EMA'] = self.EMA(self.test)
        self.test = self.Bollingerband(self.test)
        self.test = self.MACD(self.test)
        self.test = self.Momentum(self.test)
        self.test = self.RSI(self.test)
        
        self.test.dropna(inplace=True)
        self.test.reset_index(inplace=True)
        self.test.drop(columns=['index'], inplace=True)
        
        print("MACD trading...")
        macd = self.test.copy()
        print("Checking signal...", end = ' ')
        if len(macd):
            macd['signal'] = ''
            macd = self.trading(macd)
        if len(macd)!=0:
            macd.loc[0, 'signal'] = ''
            macd.loc[len(macd)-1, 'signal'] = 'end'
        print("Actual trading...", end = ' ')
        macd = self.returns(macd)
        self.test['macd_acc'] = macd['acc return']
        
        #self.test = self.break_point(self.test)
        #move = self.test['break point']
        #self.test.drop(columns = ['break point'], inplace=True)
        #self.test['break point'] = move
        
        #self.test['rl_predict'] = np.nan
        
        print("Done!")
        
        return self.train, self.test