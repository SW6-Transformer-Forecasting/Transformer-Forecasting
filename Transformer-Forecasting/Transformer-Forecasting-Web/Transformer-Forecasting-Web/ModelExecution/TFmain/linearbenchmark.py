import pandas
import numpy
# from DataHandling.dataFilter import DataFilter
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings("ignore")

dataframe = pandas.read_csv(".\Data\ETTh1OutliersRemoved.csv")


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


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 44)

model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

x_test = OTScaler.inverse_transform(x_test)
y_test = OTScaler.inverse_transform(y_test.reshape(-1,1))
predictions = OTScaler.inverse_transform(predictions.reshape(-1,1))

for i in range(len(predictions)):
    print(f"{y_test[i]} and {predictions[i]}")

print('MSE %: ', mean_absolute_percentage_error(y_test, predictions))


