import pandas
from sklearn.preprocessing import MinMaxScaler
import numpy
import os
import time


class TransformData:
    scaler = MinMaxScaler()
    
    def __init__(self):
        print("Normalizing data...")
            
        
    # Fits the scaler to the data and transforms it 
    def NormalizeDataToCSV(self, data):
        # Avoids attempting to normalize the 'date' value, which cant be converted to a float (ex. 2016-03-05 13:00:00)
        columnsToNormalize = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']
        data['datetime'] = pandas.to_datetime(data['date'])
        data[columnsToNormalize] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, 
                                                                         "hour":row.datetime.hour, "weekday":row.datetime.weekday(), 
                                                                         "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 
                                                                         'OilTemp': row.OT}), axis=1)
        
        dataframe = pandas.DataFrame(data, columns=['date', 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp'])
        dataframe[columnsToNormalize] = self.scaler.fit_transform(dataframe[columnsToNormalize])
        pandas.DataFrame(dataframe, columns=['date', 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']
                         ).to_csv('minmax_normalized_linear.csv')
        
        
    # Transforms the input based on the fit value of the scaler
    def NormalizeInput(self, data):
        dataframe = pandas.DataFrame(data, columns=['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp'])
        return self.scaler.transform(dataframe)
    
    def InverseNormalization(scaled_data):
        print("Inversing data normalization...")
        data = pandas.read_csv(".\Data\ETTh1.csv")
        columnsToNormalize = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']
        data['datetime'] = pandas.to_datetime(data['date'])
        data[columnsToNormalize] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, 
                                                                         "hour":row.datetime.hour, "weekday":row.datetime.weekday(), 
                                                                         "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 
                                                                         'OilTemp': row.OT}), axis=1)
        
        dataframe = pandas.DataFrame(data, columns=['date', 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp'])
        
        scaler2 = MinMaxScaler().fit(dataframe[columnsToNormalize]) 
        reversed_data = scaler2.inverse_transform(scaled_data)
        print(reversed_data)
        return reversed_data
        
        
data = pandas.read_csv(".\Data\ETTh1.csv")
test = TransformData(data, True)
    
        
        
        



