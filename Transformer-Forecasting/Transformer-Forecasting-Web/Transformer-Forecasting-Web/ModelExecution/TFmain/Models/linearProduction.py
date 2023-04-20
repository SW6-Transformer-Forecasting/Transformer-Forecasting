from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn import linear_model

class LinearRegression:
    def __init__(self, periodDescription, x_train, y_train, x_predict):
        linearModel = self.TrainLinearModel(x_train, y_train)
        predictedOT = linearModel.predict(x_predict)

        print(f"{periodDescription}: \n {predictedOT}")


    def TrainLinearModel(self, x_train, y_train):
        linearModel = linear_model.LinearRegression()
        return linearModel.fit(x_train, y_train)
    