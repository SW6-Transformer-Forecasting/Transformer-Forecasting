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
    def FitAndTransformData(self, dataframe):
        dataframe= self.scaler.fit_transform(dataframe)
        return dataframe
          
    def TransformData(self, dataframe):
        dataframe = self.scaler.transform(dataframe)
        return dataframe
        
    def InverseOT(self, scaled_data, OT_index, pytorch):
        print("Inversing data normalization...")

        scaled_data = self.GiveDataframeEmptyValues(scaled_data)
        
        reversed_data = self.scaler.inverse_transform(scaled_data)
        
        reversed_OT = reversed_data[:, OT_index]
        
        return reversed_OT
    
    def InverseDates(self, scaled_data):
        print("Inversing dates...")
        reversed_data = self.scaler.inverse_transform(scaled_data)
        
        reversed_dates = reversed_data[:, [0, 1, 2, 3]]
        
        reversed_dates = pandas.DataFrame(reversed_dates, columns=["month", "day", "hour", "weekday"])
        return reversed_dates
    
    # non-broadcastable output operand with shape (5,1) doesn't match the broadcast shape (5,8)
    def GiveDataframeEmptyValues(self, dataframe):
        return dataframe.apply(lambda row: pandas.Series({"month": 0.0,
                                                          "day": 0.833333, 
                                                          "hour": 0.608696, 
                                                          "weekday": 0.5,
                                                          "OT": row.OT}), axis=1)
    
    def getScaler(self):
        return self.scaler