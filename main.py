import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error     

data = pd.read_csv("ETTh1.csv")

# Makes the date on the X-axis more readable
##data.index = pd.to_datetime(data.index)

# Marks the last 3 months of the dataset as a test set
##train = data.loc[data.index < '2018-03-26 23:00:00']
##test = data.loc[data.index >= '2018-03-26 23:00:00']

# Marks the training and test data with different colors
##fig, ax = plt.subplots(figsize=(15,5))
##train.plot(ax=ax, label='Training Set')
##test.plot(ax=ax, label='Test Set')


#separate the other attributes from the predicting attribute
x = data.drop(['OT', 'date'], axis=1)
#print(x)

#separate the predicting attribute into Y for model training 
y = data['OT']

#data['OT'].plot(kind = 'line')


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)



LRmodel = LinearRegression()
LRmodel.fit(x_train, y_train)

y_prediction = LRmodel.predict(x_test)


plt.scatter(range(len(y_test)), y_test, color='blue')
plt.scatter(range(len(y_prediction)), y_prediction, color='red')



score = r2_score(y_test, y_prediction)
print('r2 score: ', score)
print('MSE: ', mean_squared_error(y_test, y_prediction))
print('MAE: ', mean_absolute_error(y_test, y_prediction))


# Adds a line between the training and test data
##ax.axvline('2018-03-26 23:00:00', color='black', ls='--')
##ax.legend(['Training Set', 'Test Set'])

#data.plot(style='.', figsize=(15,5), title='weewoo')

plt.show()


