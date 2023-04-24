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
        # 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OT'
        columnsToNormalize = ['month', 'day', 'hour', 'OT']
        # print(dataframe)
        dataframe[columnsToNormalize] = self.scaler.fit_transform(dataframe[columnsToNormalize])
        
        print(dataframe)
        return dataframe
          
        
    # Transforms the input based on the fit value of the scaler
    def NormalizeInput(self, dataframe):
        return self.scaler.transform(dataframe)
    
    def InverseNormalization(self, scaled_data):
        print("Inversing data normalization...")
    
        scaled_data = self.GiveDataframeEmptyValues(scaled_data)
        print(scaled_data)
        reversed_data = self.scaler.inverse_transform(scaled_data)
        print(reversed_data)
        return reversed_data
    
    # non-broadcastable output operand with shape (5,1) doesn't match the broadcast shape (5,8)
    def GiveDataframeEmptyValues(self, dataframe):
        return dataframe.apply(lambda row: pandas.Series({"month": 0.454545, "day": 0.833333, 
                                                                    "hour": 0.608696, 
                                                                    'OT': row.OT}), axis=1)
    
    def GiveDataframeEmptyValues2(self, dataframe):
        return dataframe.apply(lambda row: pandas.Series({"month": 0.454545, "day": 0.833333, 
                                                                    "hour": 0.608696, 
                                                                    'OT': row.OT}), axis=1)
        
        
    
