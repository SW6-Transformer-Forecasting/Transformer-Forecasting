from DataHandling.datahandler import DataHandler
from Models.linear import LinearRegression
from DataHandling.datatransformerproduction import TransformData

dataTransformer = TransformData()

# Picks the training/test periods specified in TrainTestPeriods.json and prepares data for Linear model training
dataHandler = DataHandler(dataTransformer)

# Uses the prepared data and creates Linear models for the prediction task
for modelData in dataHandler.trainTestPeriods:
    LinearRegression(modelData.periodDescription, modelData.x_train, modelData.y_train, modelData.x_test, modelData.y_test, dataTransformer)