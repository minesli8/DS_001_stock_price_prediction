import pandas as pd
from pandas_datareader import data

class Stock:

    def __init__(self, company, source, start_time, end_time):
        self.company = company
        self.source = source
        self.start_time = start_time
        self.end_time = end_time
        self.stock_data = None
        pass

    def get_data(self):
        self.stock_data = data.DataReader(self.company, self.source, self.start_time, self.end_time)
        pass

    def add_lag_data(self, max_lag):
        for i in range(1, max_lag+1, 1):
            lag = -i
            Close_lag = self.stock_data['Close'].shift(lag)
            name = 'Close_lag' + str(-lag)
            self.stock_data[name] = Close_lag
        pass