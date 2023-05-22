import pandas
import numpy
# from DataHandling.dataFilter import DataFilter
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
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

dataframe = dataframe.to_numpy()

x = numpy.delete(normalizedDataframe, numberOfColumns, axis=1)
y = normalizedDataframe[:, numberOfColumns]
print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size = 0.2)
print(x_train)
model = linear_model.LinearRegression().fit(x_train, y_train)

predictions = model.predict(x_test)

x_test = OTScaler.inverse_transform(x_test)
y_test = OTScaler.inverse_transform(y_test.reshape(-1,1))
predictions = OTScaler.inverse_transform(predictions.reshape(-1,1))

for i in range(len(predictions)):
    print(f"{y_test[i]} and {predictions[i]}")

print('MAE: ', mean_absolute_error(y_test, predictions))
print('MAPE: ', mean_absolute_percentage_error(y_test, predictions))

plt.plot(y_test)
plt.plot(predictions)
plt.show()

