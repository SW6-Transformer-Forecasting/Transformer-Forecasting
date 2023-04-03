from Models.linear import LinearRegression
from DataHandling.linearmodeldata import ModelData
import pandas
import json
import numpy
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

data = pandas.read_csv("DataHandling\ETTh1.csv")

class DataHandler:
    trainTestPeriods = []
    
    def __init__(self):
        self.GetTrainTestPeriods()
        self.SetupTrainTestData()
        
    def GetTrainTestPeriods(self):
        with open('DataHandling\TrainTestPeriods.json') as trainTestPeriodsFile:
            file_contents = trainTestPeriodsFile.read()
            parsedJson = json.loads(file_contents) 
            for period in parsedJson['ModelPeriods']:
                self.trainTestPeriods += [[period['periodDescription'], period['trainFrom'], period['trainTo'], period['testFrom'], period['testTo']]]
            
    def SetupTrainTestData(self):
        for index in range(len(self.trainTestPeriods)):
            periodDescription = self.trainTestPeriods[index][0]
                      
            data['datetime'] = pandas.to_datetime(data['date'])
                      
            trainPeriod = data[(data['date'] >= self.trainTestPeriods[index][1]) & (data['date'] < self.trainTestPeriods[index][2])]
            testPeriod = data[(data['date'] >= self.trainTestPeriods[index][3]) & (data['date'] <= self.trainTestPeriods[index][4])]
            
            columns = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']    
            trainPeriod[columns] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, "hour":row.datetime.hour, "weekday":row.datetime.weekday(), "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 'OilTemp': row.OT}), axis=1)
            testPeriod[columns] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, "hour":row.datetime.hour, "weekday":row.datetime.weekday(), "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 'OilTemp': row.OT}), axis=1)
        
            features = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter']
            target = ['OT']
            
            x_train = trainPeriod[features]
            y_train = trainPeriod[target]
            
            x_test = testPeriod[features]
            y_test = testPeriod[target]               
                    
            self.trainTestPeriods[index] = ModelData(periodDescription, x_train, y_train, x_test, y_test)
            
