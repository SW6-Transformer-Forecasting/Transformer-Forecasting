import pandas 
import numpy 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import os

# venter med feature selection til json er tilføjet :D

cwd = os.getcwd()

# dataframe = pandas.read_csv(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\ETTh1.csv")
dataframe = pandas.read_csv(cwd + "\ModelExecution\TFmain\Data\ETTh1.csv")
dataframe.drop("HUFL", inplace=True, axis=1)
dataframe.drop("HULL", inplace=True, axis=1)
dataframe.drop("MUFL", inplace=True, axis=1)
dataframe.drop("MULL", inplace=True, axis=1)
dataframe.drop("LUFL", inplace=True, axis=1)
dataframe.drop("LULL", inplace=True, axis=1)

dataframe['datetime'] = pandas.to_datetime(dataframe['date'])
dataframe = dataframe.apply(lambda row: pandas.Series({
                                                            # "year":row.datetime.year,
                                                            "month":row.datetime.month,
                                                            "day":row.datetime.day,
                                                            "hour":row.datetime.hour,
                                                            # "weekday":row.datetime.weekday(),
                                                            "quarter":row.datetime.quarter,
                                                            'OT': row.OT}), axis=1)

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
        
inversedData = scaler.inverse_transform(normalizedDataframe)

dataframe = pandas.DataFrame(inversedData, columns=['month', 'day', 'hour', 'quarter', 'OT'])

# dataframe.to_csv(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\ETTh1OutliersRemoved.csv")
dataframe.to_csv(cwd + "\ModelExecution\TFmain\Data\ETTh1OutliersRemoved.csv") 