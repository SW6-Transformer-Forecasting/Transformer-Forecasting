import pandas 
from pandas.tseries.offsets import DateOffset
import numpy 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from readJSONParams import JsonParams
import os

# venter med feature selection til json er tilføjet :D

cwd = os.getcwd()

# dataframe = pandas.read_csv(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\cleandata.csv")
dataframe = pandas.read_csv(cwd + "\ModelExecution\TFmain\Data\cleandata.csv")
dataframe.drop("HUFL", inplace=True, axis=1)
dataframe.drop("HULL", inplace=True, axis=1)
dataframe.drop("MUFL", inplace=True, axis=1)
dataframe.drop("MULL", inplace=True, axis=1)
dataframe.drop("LUFL", inplace=True, axis=1)
dataframe.drop("LULL", inplace=True, axis=1)

# selects the rows that are within the TrainingPeriodLength (default is the last 3 months of data)
dataframe['date'] = pandas.to_datetime(dataframe['date'])
current_date = dataframe['date'].max()
start_date = current_date - DateOffset(months=JsonParams.TrainingPeriodLength)
dataframe = dataframe[(dataframe['date'] > start_date) & (dataframe['date'] <= current_date)]

dataframe = dataframe.apply(lambda row: pandas.Series({
                                                            "year":row.date.year,
                                                            "month":row.date.month,
                                                            "day":row.date.day,
                                                            "hour":row.date.hour,
                                                            "weekday":row.date.weekday(),
                                                            "weekofyear": row.date.weekofyear, 
                                                            "quarter":row.date.quarter,
                                                            'OT': row.OT}), axis=1)
notIncludedDateValues = JsonParams.GetNotIncludedDateValues()
dataframe = dataframe.drop(columns=notIncludedDateValues)

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
PERCENTAGEDIFFERENCE = 60

lowerPercentageBound = 100 - PERCENTAGEDIFFERENCE
higherPercentageBound = 100 + PERCENTAGEDIFFERENCE
for i in range(len(predictions)):
    # If differenceInPercentage is 100, then the test data and prediction have the exact same value
    differenceInPercentage = abs(y[i] / predictions[i] * 100)
    if(differenceInPercentage < lowerPercentageBound or differenceInPercentage > higherPercentageBound):
        deletedRows += 1
        normalizedDataframe = numpy.delete(normalizedDataframe, i - deletedRows, 0)
        
inversedData = scaler.inverse_transform(normalizedDataframe)

dateValues = JsonParams.GetIncludedDateValues()
dataframe = pandas.DataFrame(inversedData, columns=dateValues + ["OT"])

# dataframe.to_csv(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\ETTh1OutliersRemoved.csv")
dataframe.to_csv(cwd + "\ModelExecution\TFmain\Data\ETTh1OutliersRemoved.csv") 