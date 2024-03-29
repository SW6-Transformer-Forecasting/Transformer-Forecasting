import pandas
import numpy
# from DataHandling.dataFilter import DataFilter
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

dataframe = pandas.read_csv(".\Data\ETTh1OutliersRemoved.csv")
# dataframe.drop(columns=dataframe.columns[0], axis=1, inplace=True)

DAYS_TO_TRAIN_ON = 90 * 24
HOURS_TO_PREDICT_AHEAD = 72

numberOfColumns = dataframe.shape[1] - 1
x_columns = []
for i in range(numberOfColumns):
    x_columns.append(i)

scaler = MinMaxScaler()
normalizedDataframe = scaler.fit_transform(dataframe)

# saving the scaler for the oil temperature for later use when it has to be inversed
OTScaler = MinMaxScaler()
OTScaler.min_,OTScaler.scale_=scaler.min_[numberOfColumns],scaler.scale_[numberOfColumns]

x = numpy.delete(normalizedDataframe, numberOfColumns, axis=1)
y = normalizedDataframe[:, numberOfColumns]

#row != 0 and row % DAYS_TO_TRAIN_ON == 0
startTrain = 0
startTest = 0
endTrain = 0
endTest = 0
mapeList = []
trainingThreshold = len(x) - DAYS_TO_TRAIN_ON
for row in range(len(x)):
    if(row > DAYS_TO_TRAIN_ON and row < trainingThreshold):
        endTrain = row # endtrain = 480
        x_train = x[startTrain:endTrain] #start = 266
        y_train = y[startTrain:endTrain]
        
        # plt.plot(y_train)
        # plt.show()
        
        startTest = endTrain + 1
        endTest = startTest + HOURS_TO_PREDICT_AHEAD
        
        x_test = x[startTest:endTest]
        y_test = y[startTest:endTest]
        
        model = LinearRegression()
        model.fit(x_train, y_train)
        # model = removeOutliers(x_train, y_train)

        predictions = model.predict(x_test)

        x_test = OTScaler.inverse_transform(x_test)
        y_test = OTScaler.inverse_transform(y_test.reshape(-1,1))
        predictions = OTScaler.inverse_transform(predictions.reshape(-1,1))

        # plt.plot(y_test)
        # plt.plot(predictions)
        # plt.show()

        # mape = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        mapeList += [mape]
        print('MAPE: ', mape)
        # if(mape > 0):
        #     for i in range(len(predictions)):
        #         print(f"{y_test[i]} and {predictions[i]}")
        
        
        
        startTrain = endTrain
        

errorSum = 0       
for i in range(len(mapeList)):
    errorSum += mapeList[i]

# This will throw a division by zero error if there isnt much data in ETTh1OutliersRemoved
print(f"Average MAPE: {errorSum / len(mapeList)}")