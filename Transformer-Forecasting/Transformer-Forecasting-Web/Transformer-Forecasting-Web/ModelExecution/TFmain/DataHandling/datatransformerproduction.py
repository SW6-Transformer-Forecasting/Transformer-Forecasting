import pandas
from sklearn.preprocessing import MinMaxScaler
import numpy
import os
import time


class TransformData:
    scaler = MinMaxScaler()
    OTScaler = MinMaxScaler()
    
    def __init__(self):
        print("Normalizing data...")

        
    # Fits the scaler to the data and transforms it 
    def FitAndTransformData(self, dataframe):
        dataframe= self.scaler.fit_transform(dataframe)
        return dataframe
          
    def TransformData(self, dataframe):
        dataframe = self.scaler.transform(dataframe)
        return dataframe
        
    def SaveOTScaler(self, numberOfColumns):
        self.OTScaler.min_,self.OTScaler.scale_=self.scaler.min_[numberOfColumns],self.scaler.scale_[numberOfColumns]
        
    def InverseOT(self, scaled_data):
        print("Inversing data normalization...")
        reversed_OT = self.OTScaler.inverse_transform(scaled_data)
        return reversed_OT
    
    def InverseDates(self, scaled_data):
        print("Inversing dates...")
        reversed_data = self.scaler.inverse_transform(scaled_data)
        
        reversed_dates = reversed_data[:, [0, 1, 2, 3]]
        
        reversed_dates = pandas.DataFrame(reversed_dates, columns=["month", "day", "hour", "quarter"])
        return reversed_dates
    
class PytorchTransformer:
    load_scaler = MinMaxScaler()
    OT_scaler = MinMaxScaler()
    
    def fit_transform_loads(self, loads):
        return self.load_scaler.fit_transform(loads)
    
    def fit_transform_OT(self, OT):
        return self.OT_scaler.fit_transform(OT)
    
    def transform_loads(self, loads):
        return self.load_scaler.transform(loads)
    
    def transform_OT(self, OT):
        return self.OT_scaler.transform(OT)
    
    def inverse_loads(self, loads):
        return self.load_scaler.inverse_transform(loads)
    
    def inverse_OT(self, OT):
        return self.OT_scaler.inverse_transform(OT)