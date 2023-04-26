from DataHandling.linearmodeldata import ModelData
import pandas
import numpy
from sklearn.model_selection import train_test_split
from DataHandling.datatransformerproduction import TransformData
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

dataframe = pandas.read_csv(".\Data\ett3.csv")
dataframe.drop("HUFL", inplace=True, axis=1)
dataframe.drop("HULL", inplace=True, axis=1)
dataframe.drop("MUFL", inplace=True, axis=1)
dataframe.drop("MULL", inplace=True, axis=1)
dataframe.drop("LUFL", inplace=True, axis=1)
dataframe.drop("LULL", inplace=True, axis=1)

dataframe['datetime'] = pandas.to_datetime(dataframe['date'])
dataframe = dataframe.apply(lambda row: pandas.Series({
                                                                    "day": row.datetime.day,
                                                                    "month": row.datetime.month, 
                                                                    "hour":row.datetime.hour,
                                                                    "weekday":row.datetime.weekday(),
                                                                    'OT': row.OT}), axis=1)


scaler = MinMaxScaler()
normalizedDataframe = scaler.fit_transform(dataframe)

# saving the scaler for the oil temperature for later use when it has to be inversed
OTScaler = MinMaxScaler()
OTScaler.min_,OTScaler.scale_=scaler.min_[4],scaler.scale_[4]

# [:, 0] means 'selecting the first column' and so forth..
X_values = normalizedDataframe[:, [0, 1, 2, 3]]
y_values = normalizedDataframe[:, [4]]

X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=0.20, random_state=42)   

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

inversedPrediction = numpy.round(OTScaler.inverse_transform(predictions), 2)
inversedYtest = numpy.round(OTScaler.inverse_transform(y_test), 2)

# index = 0
# for thing in y_test:
#     print(f"{inversedYtest[index][0]} and {inversedPrediction[index][0]}")
#     index +=1


index = 0
for thing in y_test:
    print(f"{y_test[index][0]} and {predictions[index][0]}")
    index +=1

print('MSEinversed: ', mean_squared_error(inversedYtest, inversedPrediction))
print('MSE: ', mean_squared_error(y_test, predictions))
print('MAE inversed: ', mean_absolute_error(inversedYtest, inversedPrediction))
