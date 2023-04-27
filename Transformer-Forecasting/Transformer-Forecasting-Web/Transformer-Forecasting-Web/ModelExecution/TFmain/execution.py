from DataHandling.datahandler import DataHandler
from Models.linear import LinearRegression



# Picks the training/test periods specified in TrainTestPeriods.json and prepares data for Linear model training
dataHandler = DataHandler()


periodDescription = dataHandler.dataSplit[0]
x_train = dataHandler.dataSplit[1]
y_train = dataHandler.dataSplit[2]
x_test = dataHandler.dataSplit[3]
y_test = dataHandler.dataSplit[4]

LinearRegression(periodDescription, x_train, y_train, x_test, y_test, dataHandler.dataTransformer)

# Uses the prepared data and creates Linear models for the prediction task
# for modelData in dataHandler.trainTestPeriods:
#     LinearRegression(modelData.periodDescription, modelData.x_train, modelData.y_train, modelData.x_test, modelData.y_test, dataTransformer)