from DataHandling.linearmodeldataproduction import ModelDataProduction
from DataHandling.readJSONParams import JsonParams
import pandas
import numpy
import warnings
from datetime import datetime, timedelta
import os

warnings.filterwarnings("ignore")

cwd = os.getcwd()

dataframe = pandas.read_csv(cwd + '\ModelExecution\TFmain\Data\ETTh1.csv')
dataframe['date'] = pandas.to_datetime(dataframe['date'])
current_date = str(dataframe['date'].max())

class DataHandler:
    trainDataInformation = []
    predictDataAsDatetime = []
    linearModelInformation = []
    dateValues = JsonParams.GetIncludedDateValues()
    
    def __init__(self, periodDescription, dataTransformer):
        self.SetupTrainPredictData(periodDescription, dataTransformer)
    
    def ConvertStringToDatetime(self, dateString):
        return datetime.strptime(dateString, "%Y-%m-%d %H:%M:%S")
            
    def SetupTrainPredictData(self, periodDescription, dataTransformer):    
        # gets the training data from the outlier-filtered dataset
        trainingData = pandas.read_csv(cwd + "\ModelExecution\TFmain\Data\ETTh1OutliersRemoved.csv", usecols=self.dateValues + ["OT"])
        numberOfColumns = trainingData.shape[1] - 1
        x_columns = []
        for i in range(numberOfColumns):
            x_columns.append(i)
            
        normalizedTrainingData = dataTransformer.FitAndTransformData(trainingData)

        # Possible features: 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
        
        x = numpy.delete(normalizedTrainingData, numberOfColumns, axis=1)
        y = normalizedTrainingData[:, numberOfColumns]

        # by default this converts the next 24 hours into an input for the model
        datePredictValues = self.GetDatePredictValues(current_date)

        x_predict = pandas.DataFrame(datePredictValues)
        x_predict.columns = ['date']
        x_predict = self.ApplyDateValuesSplit(x_predict)
        x_predict = dataTransformer.TransformData(x_predict)
        dataTransformer.SaveOTScaler(numberOfColumns)

        # deletes the OT column
        x_predict = numpy.delete(x_predict, numberOfColumns, 1)

        self.linearModelInformation = ModelDataProduction(periodDescription, x, y, x_predict)
        
        
    # Sets the input dates up for normalization and prediction
    def ApplyDateValuesSplit(self, data):
        data['datetime'] = pandas.to_datetime(data['date'])
        dateSplit = data.apply(lambda row: pandas.Series({"year": row.datetime.year,
                                                          "month":row.datetime.month,
                                                          "day":row.datetime.day, 
                                                          "hour":row.datetime.hour, 
                                                          "weekday": row.datetime.weekday(),
                                                          "weekofyear": row.datetime.weekofyear,  
                                                          "quarter":row.datetime.quarter,
                                                          "OT": 0}), axis=1)
        notIncludedDateValues = JsonParams.GetNotIncludedDateValues()
        dateSplit = dateSplit.drop(columns=notIncludedDateValues)
        return dateSplit


    # Loops through the start date to the end date hourly and returns an array of their string representations
    def GetDatePredictValues(self, dateToPredictFrom):
        dateToPredictFrom = self.ConvertStringToDatetime(dateToPredictFrom)
        dates = []
        for i in range(JsonParams.HoursToPredict):
            dateToPredictFrom += timedelta(hours=1)
            dates += [str(dateToPredictFrom)]
            
        return dates
        