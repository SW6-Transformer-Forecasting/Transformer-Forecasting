from DataHandling.linearmodeldata import ModelData
import pandas
import json
import warnings

warnings.filterwarnings("ignore")

data = pandas.read_csv(".\Data\ETTh1.csv")

class DataHandler:
    trainTestPeriods = []
    
    def __init__(self, dataTransformer):
        self.GetTrainTestPeriods()
        self.SetupTrainTestData(dataTransformer)
        
    def GetTrainTestPeriods(self):
        with open('DataHandling\TrainTestPeriods.json') as trainTestPeriodsFile:
            file_contents = trainTestPeriodsFile.read()
            parsedJson = json.loads(file_contents) 
            for period in parsedJson['ModelPeriods']:
                self.trainTestPeriods += [[period['periodDescription'], period['trainFrom'], period['trainTo'], period['testFrom'], period['testTo']]]
            
    def SetupTrainTestData(self, dataTransformer):
        for index in range(len(self.trainTestPeriods)):
            periodDescription = self.trainTestPeriods[index][0]
                      
            # retrieves the relevant data directly from the ETTh1 dataset
            trainingData = data[(data['date'] >= self.trainTestPeriods[index][1]) & (data['date'] < self.trainTestPeriods[index][2])]
            testingData = data[(data['date'] >= self.trainTestPeriods[index][3]) & (data['date'] <= self.trainTestPeriods[index][4])]
        
            # transforms the retrieved ETTh1 data into 'month,day,hour,OT' format
            trainingData = self.ApplyDateValuesSplit(trainingData)
            testingData = self.ApplyDateValuesSplit(testingData)

            trainingData = dataTransformer.NormalizeTrainingData(trainingData)
            testingData = dataTransformer.NormalizeTrainingData(testingData)

            # Possible features: 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
            # features = ['month', 'hour', 'weekday', 'weekofyear', 'quarter']
            # features = ['month', 'hour', 'weekday', 'weekofyear', 'quarter']
            features = ['month', 'day', 'hour']
            target = ['OT']
            
            x_train = trainingData[features]
            y_train = trainingData[target]
            
            x_test = testingData[features]
            y_test = testingData[target]               
                    
            self.trainTestPeriods[index] = ModelData(periodDescription, x_train, y_train, x_test, y_test)
            

        # Handles the date split depending on if the data is the training data or the input to predict on
    def ApplyDateValuesSplit(self, data):
        data['datetime'] = pandas.to_datetime(data['date'])
        return data.apply(lambda row: pandas.Series({"month":row.datetime.month, "day":row.datetime.day, 
                                                                    "hour":row.datetime.hour, 
                                                                    'OT': row.OT}), axis=1)

