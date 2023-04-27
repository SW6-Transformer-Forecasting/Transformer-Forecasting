from DataHandling.linearmodeldata import ModelData
import pandas
import json
import warnings
from sklearn.model_selection import train_test_split
from DataHandling.datatransformerproduction import TransformData
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 

warnings.filterwarnings("ignore")

data = pandas.read_csv(".\Data\ETTh1.csv")

class DataHandler:
    trainTestPeriods = []
    normalizedData = []
    dataSplit = []
    dataTransformer = TransformData()
    
    def __init__(self):
        # self.GetTrainTestPeriods()
        self.NormalizeDataset()
        self.SetupTrainTestData()
        
    def GetTrainTestPeriods(self):
        with open('DataHandling\TrainTestPeriods.json') as trainTestPeriodsFile:
            file_contents = trainTestPeriodsFile.read()
            parsedJson = json.loads(file_contents) 
            for period in parsedJson['ModelPeriods']:
                self.trainTestPeriods += [[period['periodDescription'], period['trainFrom'], period['trainTo'], period['testFrom'], period['testTo']]]
        
    # Removes the load columns -> transforms the 'date' column into seperate date values -> normalizes the whole dataset
    def NormalizeDataset(self):
        dataWithModifiedColumns = data[['date', 'OT']]
        dataWithModifiedColumns = self.ApplyDateValuesSplit(dataWithModifiedColumns)
        self.normalizedData = self.dataTransformer.FitAndTransformData(dataWithModifiedColumns)
           
           
    def SetupTrainTestData(self):
        x_values = self.normalizedData[['month', 'day', 'hour']]
        y_values = self.normalizedData['OT']
    
        x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.3, shuffle=False)               

        self.dataSplit = ["cringe", x_train, y_train, x_test, y_test]
        
        y_test = y_test.to_numpy()
        
        # self.Linear(x_train, y_train, x_test, y_test)
            
    def SetupTrainTestData2(self, dataTransformer):
        for index in range(len(self.trainTestPeriods)):
            periodDescription = self.trainTestPeriods[index][0]
                      
            # retrieves the relevant data directly from the ETTh1 dataset
            
            
            # trainingData = data[(data['date'] >= self.trainTestPeriods[index][1]) & (data['date'] < self.trainTestPeriods[index][2])]
            # testingData = data[(data['date'] >= self.trainTestPeriods[index][3]) & (data['date'] <= self.trainTestPeriods[index][4])]
            
            
            x_values = self.normalizedData[['month', 'day', 'hour']]
            y_values = self.normalizedData['OT']
        
            x_train, x_test, y_train, y_test = train_test_split(x_values, y_values, random_state=104, test_size=0.25)

            # # transforms the retrieved ETTh1 data into 'month,day,hour,OT' format
            # trainingData = self.ApplyDateValuesSplit(trainingData)
            # testingData = self.ApplyDateValuesSplit(testingData)

            # trainingData = dataTransformer.FitAndTransformData(trainingData)
            # testingData = dataTransformer.TransformData(testingData)


            # # Possible features: 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
            # features = ['month', 'hour', 'weekday', 'weekofyear', 'quarter']
            # features = ['month', 'hour', 'weekday', 'weekofyear', 'quarter']
            features = ['month', 'day', 'hour']
            target = ['OT']
            
            # x_train = trainingData[features]
            # # print(x_train)
            # y_train = trainingData[target]
            
            # x_test = testingData[features]

            # y_test = testingData[target]               

            self.trainTestPeriods[index] = ModelData(periodDescription, x_train, y_train, x_test, y_test)
            

        # Handles the date split depending on if the data is the training data or the input to predict on
    def ApplyDateValuesSplit(self, data):
        data['datetime'] = pandas.to_datetime(data['date'])
        return data.apply(lambda row: pandas.Series({"month":row.datetime.month, "day":row.datetime.day, 
                                                                    "hour":row.datetime.hour, 
                                                                    'OT': row.OT}), axis=1)
        
    def Linear(self, x_train, y_train, x_test, y_test):
        linearModel = self.TrainLinearModel(x_train, y_train)
        predictedOT = linearModel.predict(x_test)

        print(type(y_test))
        print(type(predictedOT))

        # y_test = pandas.DataFrame(y_test, columns=["OT"])
        predictedOT = pandas.DataFrame(predictedOT, columns=["OT"])
        
        y_test = y_test.to_frame()
        

        print(type(y_test))
        y_test = self.dataTransformer.InverseNormalization(y_test)
        predictedOT = self.dataTransformer.InverseNormalization(predictedOT)
        
        self.PrintLoss(y_test, predictedOT)
        
    def TrainLinearModel(self, x_train, y_train):
        linearModel = linear_model.LinearRegression()
        return linearModel.fit(x_train, y_train)
    
    def PrintLoss(self, y_test, predictedOT):
        print('MSE: ', mean_squared_error(y_test, predictedOT))
        print('MAE: ', mean_absolute_error(y_test, predictedOT))
        print('\n')

