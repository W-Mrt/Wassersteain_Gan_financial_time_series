import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
#---------trading metrics
import stockstats
import talib as ta
#---------local modules
import stats

warnings.filterwarnings('ignore')

def k_stochastic_oscillator(df, periods=14):
    copy = df.copy()
    
    high_roll = copy["High"].rolling(periods).max()
    low_roll = copy["Low"].rolling(periods).min()
    
    # Fast stochastic indicator
    num = copy["Close"] - low_roll
    denom = high_roll - low_roll
    copyk = (num / denom) * 100
    
    return copyk

def d_stochastic_oscillator(df, periods=14):
    copy = df.copy()
    
    high_roll = copy["High"].rolling(periods).max()
    low_roll = copy["Low"].rolling(periods).min()
    
    # Fast stochastic indicator
    num = copy["Close"] - low_roll
    denom = high_roll - low_roll
    copy["%K"] = (num / denom) * 100

    # Slow stochastic indicator
    copyd = copy["%K"].rolling(3).mean()
    
    return copyd

class data_manager():
	def __init__(self):
		self.csv_file = ''
		#pass

#where to normalize?
	def prepare_pd(self):
		self.read_csv()
		self.parse_date()
		self.data_begin_date = None
		self.data_end_date = None
		self.add_columns()

	def add_columns(self,is_financial = True, contain_adjusted = True,contain_volume=True,validation_split=0.81):
		if is_financial:
			if contain_adjusted:
				self.data['Log Return'] = np.log(self.data['Adj Close']) - np.log(self.data['Adj Close'].shift(1))
				self.data['Pct Change'] = self.data['Adj Close'].pct_change().dropna()
			else:
				self.data['Log Return'] = np.log(self.data['Close']) - np.log(self.data['Close'].shift(1))
				self.data['Pct Change'] = self.data['Close'].pct_change().dropna()
			self.data['U'] = np.log(self.data['High']/self.data['Open'])
			self.data['D'] = np.log(self.data['Low']/self.data['Open'])
			self.data['C'] = np.log(self.data['Close']/self.data['Open'])
			self.data['A'] = self.data.apply(lambda x: 1 if x['Log Return'] >=0 else 0 ,axis=1)
			self.data['Y'] = self.data['A'].shift(-1)
			self.data['Daily Volatility'] = np.sqrt(0.511*((self.data['U']-self.data['D'])**2) - 0.019*(self.data['C']*(self.data['U']+self.data['D'])-2*self.data['U']*self.data['D']) - 0.383*(self.data['C']**2))
			self.data['Simple 10-day MA'] = pd.Series(self.data['Close']).rolling(window=10).mean()
			self.data['Weighted 10-day MA'] = ta.WMA(np.array(self.data['Close'],dtype=float),timeperiod=10)
			self.data['Momentum'] = ta.MOM(np.array(self.data['Close'],dtype=float), timeperiod=10)
			self.data['Stochastic K'] = k_stochastic_oscillator(self.data, 14)
			self.data['Stochastic D'] = d_stochastic_oscillator(self.data, 14)
			self.data['sign'] = self.data.apply(lambda x: 1 if x['Log Return'] >=0 else -1 ,axis=1)
			self.data['RSI'] = ta.RSI(np.array(self.data['Close'],dtype=float), timeperiod=14)
			self.data['MACD'] = ta.MACD(np.array(self.data['Close'],dtype=float),fastperiod=12, slowperiod=26, signalperiod=9)[0]
			self.data['Larry William R'] = ta.WILLR(np.array(self.data['High'],dtype=float), np.array(self.data['Low'],dtype=float),np.array(self.data['Close'],dtype=float), timeperiod=14)
			self.data['CCI'] = ta.CCI(np.array(self.data['Close'],dtype=float), np.array(self.data['Low'],dtype=float), np.array(self.data['High'],dtype=float), timeperiod=20)
			if contain_volume:
				self.data['Mean Volume 300'] = pd.Series(self.data['Volume']).rolling(window=300).mean()
				self.data['Std Volume 300'] = pd.Series(self.data['Volume']).rolling(window=300).std()
				self.data['Normalized Volume 300'] = (self.data['Volume']-self.data['Mean Volume 300'])/self.data['Std Volume 300']
			self.data['Stdev2'] = pd.Series(self.data['Log Return']).rolling(window=2).std()
			self.data['Stdev7'] = pd.Series(self.data['Log Return']).rolling(window=7).std()
			self.data['Stdev14'] = pd.Series(self.data['Log Return']).rolling(window=14).std()
			self.data['Stdev21'] = pd.Series(self.data['Log Return']).rolling(window=21).std()
			self.data['Stdev28'] = pd.Series(self.data['Log Return']).rolling(window=28).std()
			self.data = self.data.dropna()

	def read_csv(self):
		self.data = pd.read_csv(self.csv_file)


	def get_pd_table(self,**kwargs):
		table = self.data
		if 'date_start' in kwargs:
			
			table = table.ix[table['Date'] > kwargs['date_start']]
			kwargs.pop('date_start')
		if 'date_end' in kwargs:
		
			table = table.ix[self.data['Date'] < kwargs['date_end']]
			kwargs.pop('date_end')

		return table

	def get_dataset(self,column_list,look_back = 10,normalize_width = 0,normalize_scheme = None,train_ratio = 0.8):
		#get specified pd table
		specified_table = self.data(column_list)

		#Divide data into train/test
		train_x = None
		train_y = None
		test_x = None
		test_y = None
		return self._create_dataset(train_x,train_y),self._create_dataset(test_x,test_y)
		pass

	def create_dataset(self,x_set, y_set,look_back=10):
		data_x, data_y = [], []
		for i in range(len(x_set)-look_back-1):
			a = x_set[i:(i+look_back)]
			data_x.append(a)
			data_y.append(y_set[i + look_back - 1])
		return np.array(data_x), np.array(data_y)

	def parse_date(self):
		self.data['Date'] = pd.to_datetime(self.data['Date'])
		oldtime_table = self.data.copy()
		oldtime_table['Date'] -= pd.Timedelta(36500,'D')
		self.data = self.data.where(self.data['Date'] < pd.to_datetime('2022/1/1'),oldtime_table)

class snp500_individual(data_manager):
	def __init__(self,code):
		self.csv_file = "./data/SNP500/SNP500_Individuals/" +  str(code) + '.csv'

	def read_csv(self):
		self.data = pd.read_csv(self.csv_file,dtype={'Open': np.float32,'Low': np.float32,'High': np.float32,'Close': np.float32,'Volume': np.float32,'Adj_Close': np.float32})
		self.data.rename(columns={'Adj_Close':'Adj Close'},inplace=True)
		self.data = self.data.iloc[:,:-1]

class snp500(data_manager):
	def __init__(self):
		self.csv_file = './data/SNP500/snp500.csv'

	def get_code_list(self):
		t = pd.read_csv('./data/SNP500/SNP500_Individuals/constituents.csv')
		return t['Symbol'].values