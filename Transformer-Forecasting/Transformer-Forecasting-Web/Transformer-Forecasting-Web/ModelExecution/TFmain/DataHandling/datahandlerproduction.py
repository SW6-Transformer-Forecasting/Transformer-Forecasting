from DataHandling.linearmodeldataproduction import ModelDataProduction
from DataHandling.datatransformerproduction import TransformData
import pandas
import numpy
import json
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings("ignore")

cwd = os.getcwd()

# data = pandas.read_csv(cwd + "\ModelExecution\TFmain\DataHandling\ETTh1-linear-format.csv")
data = pandas.read_csv(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\ETTh1.csv")
data.drop("HUFL", inplace=True, axis=1)
data.drop("HULL", inplace=True, axis=1)
data.drop("MUFL", inplace=True, axis=1)
data.drop("MULL", inplace=True, axis=1)
data.drop("LUFL", inplace=True, axis=1)
data.drop("LULL", inplace=True, axis=1)


EARLIEST_TRAIN_DATE = "2016-07-01 00:00:00"

class DataHandler:
    trainDataInformation = []
    predictDataAsDatetime = []
    linearModelInformation = []
    
    def __init__(self, periodDescription, startPredictDate, endPredictDate, dataTransformer):
        self.GetTrainPeriodData(startPredictDate)
        self.ConvertPredictValuesToDatetime(startPredictDate, endPredictDate)
        self.SetupTrainPredictData(periodDescription, dataTransformer)

    def ConvertPredictValuesToDatetime(self, startPredictDate, endPredictDate):
        startPredictAsDatetime = self.ConvertStringToDatetime(startPredictDate)
        endPredictDateAsDatetime = self.ConvertStringToDatetime(endPredictDate)

        self.predictDataAsDatetime = [startPredictAsDatetime, endPredictDateAsDatetime]
        
    def GetTrainPeriodData(self, startPredictDate):
        trainFrom = self.SubtractOneMonthFromDate(startPredictDate)
        trainTo = startPredictDate
        self.trainDataInformation = [trainFrom, trainTo]

    def SubtractOneMonthFromDate(self, startPredictDate):
        startPredictDateMinusOneYear = self.ConvertStringToDatetime(startPredictDate) - timedelta(days=30)

        if(str(startPredictDateMinusOneYear) < EARLIEST_TRAIN_DATE):
            startPredictDate = EARLIEST_TRAIN_DATE
        
        return str(startPredictDateMinusOneYear)
    
    def ConvertStringToDatetime(self, dateString):
        return datetime.strptime(dateString, "%Y-%m-%d %H:%M:%S")
            
    def SetupTrainPredictData(self, periodDescription, dataTransformer):    

        # data['datetime'] = pandas.to_datetime(data['date'])
                    
        # gets the training data from the ETTh1 dataset
        trainingData = data[(data['date'] >= self.trainDataInformation[0]) & (data['date'] < self.trainDataInformation[1])]
        
        trainingData = self.ApplyDateValuesSplit(trainingData, True)

        # normalizes the data
        trainingData = dataTransformer.FitAndTransformData(trainingData)

        # Possible features: 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
        
        x_train = trainingData[:, [0, 1, 2, 3]]
        y_train = trainingData[:, [4]]

        datePredictValues = self.GetDatePredictValues()
        
        x_predict = pandas.DataFrame(datePredictValues)
        x_predict.columns = ['date']
        x_predict = self.ApplyDateValuesSplit(x_predict, False)

        
        x_predict = dataTransformer.TransformData(x_predict)

        x_predict = numpy.delete(x_predict, 4, 1)

        self.linearModelInformation = ModelDataProduction(periodDescription, x_train, y_train, x_predict)
        
    # Handles the date split depending on if the data is the training data or the input to predict on
    def ApplyDateValuesSplit(self, data, isTrainingData):
        data['datetime'] = pandas.to_datetime(data['date'])
        if(isTrainingData):
            return data.apply(lambda row: pandas.Series({"month":row.datetime.month, 
                                                         "day":row.datetime.day, 
                                                         "hour":row.datetime.hour, 
                                                         "weekday":row.datetime.weekday(),
                                                         "OT": row.OT}), axis=1)
        else:
            return data.apply(lambda row: pandas.Series({"month":row.datetime.month, 
                                                         "day":row.datetime.day, 
                                                         "hour":row.datetime.hour, 
                                                         "weekday":row.datetime.weekday(),
                                                         "OT": 0}), axis=1)


    # Loops through the start date to the end date hourly and returns an array of their string representations
    def GetDatePredictValues(self):
        startPredictDate = self.predictDataAsDatetime[0]
        endPredictDate = self.predictDataAsDatetime[1]

        datePredictValues = []
        hour_delta = timedelta(hours=1)

        while startPredictDate <= endPredictDate:
            datePredictValues += [str(startPredictDate)]
            startPredictDate += hour_delta

        

        return datePredictValues