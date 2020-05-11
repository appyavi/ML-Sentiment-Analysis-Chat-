import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import re
# np.warnings.filterwarnings('ignore')
# a = pd.read_excel('Retail_and_Food_Services_Monthly_Sales_Data.xls')


class MODEL_ARIMA:
    def __init__(self, start_month, end_month):
        a = pd.read_excel('Retail_and_Food_Services_Monthly_Sales_Data.xls')
        self.data = a['Total_Revenue']
        self.train_data = a['Total_Revenue'][start_month:end_month]
        self.start = start_month
        self.end = end_month

    def train(self):
        self.model = ARIMA(self.train_data, order=(1, 1, 1))
        self.model_fit = self.model.fit(disp=False)

    def predict(self, number_of_month):
        self.number_of_month = number_of_month
        yhat = self.model_fit.predict(
            len(self.train_data), len(self.train_data) + number_of_month - 1, typ='levels')
        return yhat

    def accuracy_error(self, result):
        original_data = self.data[self.end:self.end+self.number_of_month]
        error = np.linalg.norm(original_data-result) / len(result)
        return error


class MODEL_HWES:
    def __init__(self, start_month, end_month):
        a = pd.read_excel('Retail_and_Food_Services_Monthly_Sales_Data.xls')
        self.data = a['Total_Revenue']
        self.train_data = a['Total_Revenue'][start_month:end_month]
        out_vec = self.train_data
        self.start = start_month
        self.end = end_month

    def train(self):
        self.model = ExponentialSmoothing(np.asarray(self.train_data))
        self.model_fit = self.model.fit()

    def predict(self, number_of_month):
        self.number_of_month = number_of_month
        yhat = self.model_fit.predict(
            len(self.train_data), len(self.train_data) + number_of_month - 1)
        return yhat

    def accuracy_error(self, result):
        original_data = self.data[self.end:self.end+self.number_of_month]
        error = np.linalg.norm(original_data-result) / len(result)
        return error


arimamodel = MODEL_ARIMA(0, 200)
arimamodel.train()
yhat = arimamodel.predict(100)
arima_accuracy = arimamodel.accuracy_error(yhat)


hwesmodel = MODEL_HWES(0, 200)
hwesmodel.train()
yhat = hwesmodel.predict(100)
hwes_accuracy = hwesmodel.accuracy_error(yhat)
if arima_accuracy < hwes_accuracy:
    print('ARIMA gives less error.')
else:
    print('HWES gives less error.')
