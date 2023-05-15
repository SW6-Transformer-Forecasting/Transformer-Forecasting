import pandas
import numpy
# from DataHandling.dataFilter import DataFilter
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

cwd = os.getcwd()

# data = DataFilter.fetch(None, cwd + "\Data\ETTh1.csv", "2018-06-19 19:00:00", "2018-06-26 19:00:00")
# DataFilter.execute(None, data, r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web")
# dataframe = pandas.read_csv(".\Data\cleandata.csv")
dataframe = pandas.read_csv(".\Data\ETThTrain.csv")
dataframe.drop("HUFL", inplace=True, axis=1)
dataframe.drop("HULL", inplace=True, axis=1)
dataframe.drop("MUFL", inplace=True, axis=1)
dataframe.drop("MULL", inplace=True, axis=1)
dataframe.drop("LUFL", inplace=True, axis=1)
dataframe.drop("LULL", inplace=True, axis=1)

dataframe2 = pandas.read_csv(".\Data\ETThTest.csv")
dataframe2.drop("HUFL", inplace=True, axis=1)
dataframe2.drop("HULL", inplace=True, axis=1)
dataframe2.drop("MUFL", inplace=True, axis=1)
dataframe2.drop("MULL", inplace=True, axis=1)
dataframe2.drop("LUFL", inplace=True, axis=1)
dataframe2.drop("LULL", inplace=True, axis=1)

# 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
dataframe['datetime'] = pandas.to_datetime(dataframe['date'])
dataframe = dataframe.apply(lambda row: pandas.Series({
                                                                    # "year":row.datetime.year,
                                                                    "month":row.datetime.month,
                                                                    "hour":row.datetime.hour,
                                                                    "day":row.datetime.day,
                                                                    # "weekday":row.datetime.weekday(),
                                                                    "quarter":row.datetime.quarter,
                                                                    'OT': row.OT}), axis=1)

dataframe2['datetime'] = pandas.to_datetime(dataframe2['date'])
dataframe2 = dataframe2.apply(lambda row: pandas.Series({
                                                                    # "year":row.datetime.year,
                                                                    "month":row.datetime.month,
                                                                    "hour":row.datetime.hour,
                                                                    "day":row.datetime.day,
                                                                    # "weekday":row.datetime.weekday(),
                                                                    "quarter":row.datetime.quarter,
                                                                    'OT': row.OT}), axis=1)

# --------------------------

numberOfColumns = dataframe.shape[1] - 1
x_columns = []
for i in range(numberOfColumns):
    x_columns.append(i)

scaler = MinMaxScaler()
normalizedDataframe = scaler.fit_transform(dataframe)
normalizedTestDataframe = scaler.transform(dataframe2)


# saving the scaler for the oil temperature for later use when it has to be inversed
OTScaler = MinMaxScaler()
OTScaler.min_,OTScaler.scale_=scaler.min_[numberOfColumns],scaler.scale_[numberOfColumns]

x = numpy.delete(normalizedDataframe, numberOfColumns, axis=1)
y = normalizedDataframe[:, numberOfColumns]

model = LinearRegression()

model.fit(x, y)

predictions = model.predict(x)

y = OTScaler.inverse_transform(y.reshape(-1, 1))
predictions = OTScaler.inverse_transform(predictions.reshape(-1, 1))

y = numpy.asarray(y)
deletedRows = 0
for i in range(len(predictions)):
    # If differenceInPercentage is 100, then the test data and prediction have the exact same value
    differenceInPercentage = abs(y[i] / predictions[i] * 100)
    if(differenceInPercentage < 40 or differenceInPercentage > 160):
        deletedRows += 1
        normalizedDataframe = numpy.delete(normalizedDataframe, i - deletedRows, 0)
    else:
        print(f"{abs(y[i] / predictions[i] * 100)} ({y[i]} and {predictions[i]})")
        

        

print(len(normalizedDataframe))


x = numpy.delete(normalizedDataframe, numberOfColumns, axis=1)
y = normalizedDataframe[:, numberOfColumns]

X_test = numpy.delete(normalizedTestDataframe, numberOfColumns, axis=1)
y_test = normalizedTestDataframe[:, numberOfColumns]

testX = dataframe2.drop("OT", axis=1)
testY = dataframe2["OT"]

model = LinearRegression()

model.fit(x, y)

predictions = model.predict(X_test)
preddf = pandas.DataFrame({"OT": predictions})

X_test = OTScaler.inverse_transform(X_test.reshape(-1,1))
y_test = OTScaler.inverse_transform(y_test.reshape(-1,1))
predictions = OTScaler.inverse_transform(predictions.reshape(-1,1))

y = numpy.asarray(y)
for i in range(len(predictions)):
    print(f"{y_test[i]} and {predictions[i]}")
    # print(abs(((y[i] - predictions[i]) * 100) / predictions[i]))

print('MSE %: ', mean_absolute_percentage_error(y_test, predictions))
