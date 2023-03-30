import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn import linear_model

data = pd.read_csv("ETTh1.csv")
data['date'] = pd.to_datetime(data['date'])

# Adds an extra column to the dataframe that displays the number of hours since the start of the data
data['date_delta'] = (data['date'] - data['date'].min()) / np.timedelta64(1, 'h')

trainPeriod = data[(data['date'] >= '2017-12-01 00:00:00') & (data['date'] < '2017-12-31 00:00:00')]
testPeriod = data.loc[(data['date'] >= '2018-01-01 00:00:00') & (data['date'] <= '2018-01-01 06:00:00')]

x_train = np.array(trainPeriod['date_delta'])
y_train = np.array(trainPeriod['OT'])

x_test = np.array(testPeriod['date_delta'])
y_test = np.array(testPeriod['OT'])

print(trainPeriod)

regr = linear_model.LinearRegression()
regr.fit(x_train.reshape(-1,1), y_train.reshape(-1,1))

predictedOT = regr.predict(x_test.reshape(-1,1))

print('MSE: ', mean_squared_error(y_test, predictedOT))
print('MAE: ', mean_absolute_error(y_test, predictedOT))







