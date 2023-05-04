import pandas
import numpy
from sklearn.model_selection import train_test_split
from DataHandling.dataFilter import DataFilter
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as p
from scipy import stats
import warnings
import os
warnings.filterwarnings("ignore")

cwd = os.getcwd()

# data = DataFilter.fetch(None, cwd + "\Data\ETTh1.csv", "2018-06-19 19:00:00", "2018-06-26 19:00:00")
# DataFilter.execute(None, data, r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web")
# dataframe = pandas.read_csv(".\Data\cleandata.csv")
dataframe = pandas.read_csv(".\Data\outliertest.csv")
dataframe.drop("HUFL", inplace=True, axis=1)
dataframe.drop("HULL", inplace=True, axis=1)
dataframe.drop("MUFL", inplace=True, axis=1)
dataframe.drop("MULL", inplace=True, axis=1)
dataframe.drop("LUFL", inplace=True, axis=1)
dataframe.drop("LULL", inplace=True, axis=1)

dataframe.plot(x="date", y="OT", kind="line")

# p.show()


dataframe = dataframe[(numpy.abs(stats.zscore(dataframe["OT"])) < 2)]
print(f"Rows: {len(dataframe)}")

# 'year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter'
dataframe['datetime'] = pandas.to_datetime(dataframe['date'])
dataframe = dataframe.apply(lambda row: pandas.Series({
                                                                    # "year":row.datetime.year,
                                                                    # "month":row.datetime.month,
                                                                    "hour":row.datetime.hour,
                                                                    # "day":row.datetime.day,
                                                                    "weekday":row.datetime.weekday(),
                                                                    # "quarter":row.datetime.quarter,
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

# [:, 0] means 'selecting the first column' and so forth..
X_values = normalizedDataframe[:, x_columns]
y_values = normalizedDataframe[:, [numberOfColumns]]

X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.2, random_state=42)   

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

inversedPrediction = numpy.round(OTScaler.inverse_transform(predictions), 2)
inversedYtest = numpy.round(OTScaler.inverse_transform(y_test), 2)

index = 0
for thing in y_test:
    print(f"{inversedYtest[index][0]} and {inversedPrediction[index][0]}")
    index +=1


# index = 0
# for x in range(y_test):
#     print(f"{y_test[index][0]} and {predictions[index][0]}")
#     index +=1

print('MSE inversed: ', mean_squared_error(inversedYtest, inversedPrediction))
print('MAE inversed: ', mean_absolute_error(inversedYtest, inversedPrediction))
print('MSE %: ', mean_absolute_percentage_error(inversedYtest, inversedPrediction))