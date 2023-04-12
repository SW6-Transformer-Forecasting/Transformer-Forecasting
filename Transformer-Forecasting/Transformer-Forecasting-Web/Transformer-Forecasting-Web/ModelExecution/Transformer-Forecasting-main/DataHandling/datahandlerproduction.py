from DataHandling.linearmodeldataproduction import ModelDataProduction
import pandas
import json
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

data = pandas.read_csv("DataHandling\ETTh1-linear-format.csv")

EARLIEST_TRAIN_DATE = "2016-07-01 00:00:00"

class DataHandler:
    trainDataInformation = []
    predictDataAsDatetime = []
    linearModelInformation = []
    
    def __init__(self, periodDescription, startPredictDate, endPredictDate):
        self.GetTrainPeriodData()
        self.GetTrainPeriodData(startPredictDate)
        self.SetupTrainPredictData(periodDescription, startPredictDate, endPredictDate)

    def ConvertPredictValuesToDatetime(self, startPredictDate, endPredictDate):
        startPredictAsDatetime = self.ConvertPredictValuesToDatetime(startPredictDate)
        endPredictDateAsDatetime = self.ConvertPredictValuesToDatetime(endPredictDate)

        self.predictDataAsDatetime = [startPredictAsDatetime, endPredictDateAsDatetime]
        
    def GetTrainPeriodData(self, startPredictDate):
        trainFrom = self.SubtractOneYearFromDate(startPredictDate)
        trainTo = startPredictDate
        self.trainDataInformation = [trainFrom, trainTo]

    def SubtractOneYearFromDate(self, startPredictDate):
        startPredictDateMinusOneYear = self.ConvertStringToDatetime(startPredictDate) - timedelta(days=365)

        if(startPredictDateMinusOneYear < EARLIEST_TRAIN_DATE):
            startPredictDate = EARLIEST_TRAIN_DATE
        
        return str(startPredictDateMinusOneYear)
    
    def ConvertStringToDatetime(self, dateString):
        return datetime.strptime(dateString, "%Y-%m-%d %H:%M:%S")
            
    def SetupTrainPredictData(self, periodDescription):    

        data['datetime'] = pandas.to_datetime(data['date'])
                    
        trainPeriod = data[(data['date'] >= self.trainDataInformation[0]) & (data['date'] < self.trainDataInformation[1])]
    
        # Possible features: 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
        features = ['month', 'day', 'hour']
        target = ['OT']
        
        x_train = trainPeriod[features]
        y_train = trainPeriod[target]

        datePredictValues = self.GetDatePredictValues()
        x_predict = pandas.DataFrame(datePredictValues)

        self.linearModelInformation = ModelDataProduction(periodDescription, x_train, y_train, x_predict)
            
    # Loops through the start date to the end date hourly and returns an array of their string representations
    def GetDatePredictValues(self):
        startPredictDate = self.predictDataAsDatetime[0]
        endPredictDate = self.predictDataAsDatetime[1]

        datePredictValues = []
        hour_delta = timedelta(hours=1)

        while startPredictDate <= endPredictDate:
            datePredictValues += str(startPredictDate)
            startPredictDate += hour_delta

        return datePredictValues