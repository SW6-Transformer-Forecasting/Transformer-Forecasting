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
    def NormalizeTrainingData(self, dataframe):
        columnsToNormalize = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OT']
        dataframe[columnsToNormalize] = self.scaler.fit_transform(dataframe[columnsToNormalize])
        
        print(dataframe)
        return dataframe
        
        
        
    # Transforms the input based on the fit value of the scaler
    def NormalizeInput(self, dataframe):
        return self.scaler.transform(dataframe)
    
    def InverseNormalization(self, scaled_data):
        print("Inversing data normalization...")
        # data = pandas.read_csv(".\Data\ETTh1.csv")
        # columnsToNormalize = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']
        # data['datetime'] = pandas.to_datetime(data['date'])
        # data[columnsToNormalize] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, 
        #                                                                  "hour":row.datetime.hour, "weekday":row.datetime.weekday(), 
        #                                                                  "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 
        #                                                                  'OilTemp': row.OT}), axis=1)
        
        # dataframe = pandas.DataFrame(data, columns=['date', 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp'])
        scaled_data = self.GiveDataframeEmptyValues(scaled_data)
        print(scaled_data)
        reversed_data = self.scaler.inverse_transform(scaled_data)
        print(reversed_data)
        return reversed_data
    
    # non-broadcastable output operand with shape (5,1) doesn't match the broadcast shape (5,8)
    def GiveDataframeEmptyValues(self, dataframe):
        return dataframe.apply(lambda row: pandas.Series({'year': 0.0,"month": 0.5, "day": 0.5, 
                                                                    "hour": 0.5, "weekday": 0.5, 
                                                                    "weekofyear": 0.5, "quarter": 0.5, 
                                                                    'OT': row.OT}), axis=1)
        
        
    
    
    
        
        

# print(os.listdir('.'))

# scaler = MinMaxScaler()

# data = pandas.read_csv("ETTh1.csv")

# data['datetime'] = pandas.to_datetime(data['date'])

# columns = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']    
# data[columns] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, "hour":row.datetime.hour, "weekday":row.datetime.weekday(), "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 'OilTemp': row.OT}), axis=1)

# df = pandas.DataFrame(data, columns=['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp'])

# normalized_Data = scaler.fit_transform(df)

# print(normalized_Data)

# pandas.DataFrame(normalized_Data, columns=['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']).to_csv('minmax_normalized_linear.csv')

