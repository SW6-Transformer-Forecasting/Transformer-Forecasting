from DataHandling.datahandlerproduction import DataHandler
from Models.linearProduction import LinearRegression
import sys

periodDescription = sys.argv[1]
startPredictDate = sys.argv[2]
endPredictDate = sys.argv[3]

print(periodDescription)
print(startPredictDate)
print(endPredictDate)

# periodDescription = "This is an example period description"
# startPredictDate = "2018-06-26 00:00:00"
# endPredictDate = "2018-06-26 19:00:00"

# Picks the training/test periods specified in TrainTestPeriods.json and prepares data for Linear model training
dataHandler = DataHandler(periodDescription, startPredictDate, endPredictDate)

# Uses the prepared data and creates Linear models for the prediction task
modelData = dataHandler.linearModelInformation
print(modelData)
LinearRegression(modelData.periodDescription, modelData.x_train, modelData.y_train, modelData.x_predict)
