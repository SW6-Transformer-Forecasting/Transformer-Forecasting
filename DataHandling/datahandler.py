from Models.linear import LinearRegression
from DataHandling.linearmodeldata import ModelData
import pandas
import json
import numpy
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


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
            
            trainPeriod = data[(data['date'] >= self.trainTestPeriods[index][1]) & (data['date'] < self.trainTestPeriods[index][2])]
            testPeriod = data.loc[(data['date'] >= self.trainTestPeriods[index][3]) & (data['date'] <= self.trainTestPeriods[index][4])]
            
            trainPeriod['date'] = pandas.to_datetime(trainPeriod['date'])
            testPeriod['date'] = pandas.to_datetime(testPeriod['date'])
            
            trainPeriod['hour'] = trainPeriod['date'].dt.hour
            testPeriod['hour'] = testPeriod['date'].dt.hour

            # Scales the value of the oil temperature
            trainPeriod['OT'] = trainPeriod['OT'].multiply(0.5)
            testPeriod['OT'] = testPeriod['OT'].multiply(0.5)

            x_train = numpy.array(trainPeriod['hour']).reshape(-1,1)
            y_train = numpy.array(trainPeriod['OT']).reshape(-1,1)

            x_test = numpy.array(testPeriod['hour']).reshape(-1,1)
            y_test = numpy.array(testPeriod['OT']).reshape(-1,1)            
            
            self.trainTestPeriods[index] = ModelData(periodDescription, x_train, y_train, x_test, y_test)

