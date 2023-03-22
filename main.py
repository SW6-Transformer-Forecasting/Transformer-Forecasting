import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("ETTh1.csv")
x = data[['HUFL', 'HULL', 'MUFL','MULL','LUFL', 'LULL']]
y = data['OT']

trainPeriod = data[(data['date'] < '2018-01-31 23:00:00') & (data['date'] >= '2018-01-01 23:00:00')]

x_train = trainPeriod[['HUFL', 'HULL', 'MUFL','MULL', 'LUFL', 'LULL']]
y_train = trainPeriod['OT']

print(x_train)
print(y_train)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

testPeriod = data.loc[(data['date'] >= '2018-02-01 23:00:00') & (data['date'] <= '2018-02-08 23:00:00')]
x_test = testPeriod[['HUFL', 'HULL', 'MUFL','MULL','LUFL', 'LULL']]
y_test = testPeriod['OT']

x_test_scaled = scaler.fit_transform(x_test)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

predictedOT = regr.predict(x_test)



# for i in range(len(predictedOT)):
#     f.write("Actual value: " + x_test[i] + "Predicted value: " + predictedOT[i])

plt.scatter(range(len(y_test)), y_test, color='blue')
plt.scatter(range(len(predictedOT)), predictedOT, color='red')


print('MSE: ', mean_squared_error(y_test, predictedOT))
print('MAE: ', mean_absolute_error(y_test, predictedOT))

plt.show()








# #separate the other attributes from the predicting attribute
# x = data.drop(['OT', 'date'], axis=1)
# #print(x)

# #separate the predicting attribute into Y for model training 
# y = data['OT']

# #data['OT'].plot(kind = 'line')


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)



# LRmodel = LinearRegression()
# LRmodel.fit(x_train, y_train)

# y_prediction = LRmodel.predict(x_test)


# plt.scatter(range(len(y_test)), y_test, color='blue')
# plt.scatter(range(len(y_prediction)), y_prediction, color='red')



# score = r2_score(y_test, y_prediction)
# print('r2 score: ', score)
# print('MSE: ', mean_squared_error(y_test, y_prediction))
# print('MAE: ', mean_absolute_error(y_test, y_prediction))


# # Adds a line between the training and test data
# ##ax.axvline('2018-03-26 23:00:00', color='black', ls='--')
# ##ax.legend(['Training Set', 'Test Set'])

# #data.plot(style='.', figsize=(15,5), title='weewoo')

# plt.show()


