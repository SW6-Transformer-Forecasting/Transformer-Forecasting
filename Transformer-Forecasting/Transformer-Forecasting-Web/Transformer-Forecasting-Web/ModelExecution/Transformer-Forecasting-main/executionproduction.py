from DataHandling.datahandlerproduction import DataHandler
from Models.linear import LinearRegression
import sys

periodDescription = sys.argv[1]
startPredictDate = sys.argv[2]
endPredictDate = sys.argv[3]

# Picks the training/test periods specified in TrainTestPeriods.json and prepares data for Linear model training
dataHandler = DataHandler(periodDescription, startPredictDate, endPredictDate)

# Uses the prepared data and creates Linear models for the prediction task
modelData = dataHandler.trainAndPredictInformation
LinearRegression(modelData.periodDescription, modelData.x_train, modelData.y_train, modelData.x_predict)
