from DataHandling.linearmodeldata import ModelData
import pandas
import json
import warnings

warnings.filterwarnings("ignore")

data = pandas.read_csv("DataHandling\ETTh1-linear-format.csv")

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
        
            # Possible features: 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
            features = ['month', 'day', 'hour']
            target = ['OT']
            
            x_train = trainPeriod[features]
            y_train = trainPeriod[target]
            
            x_test = testPeriod[features]
            y_test = testPeriod[target]               
                    
            self.trainTestPeriods[index] = ModelData(periodDescription, x_train, y_train, x_test, y_test)
            

