from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn import linear_model

class LinearRegression:
    def __init__(self, periodDescription, x_train, y_train, x_test, y_test):
        linearModel = self.TrainLinearModel(x_train, y_train)
        predictedOT = linearModel.predict(x_test)

        for x in range(len(y_test)):
            print(f"Test: {y_test[x]} Predicted: {predictedOT[x]} Difference: {abs(y_test[x]-predictedOT[x])}")

        self.PrintLoss(periodDescription, y_test, predictedOT)

    def TrainLinearModel(self, x_train, y_train):
        linearModel = linear_model.LinearRegression()
        return linearModel.fit(x_train, y_train)
    
    def PrintLoss(self, periodDescription, y_test, predictedOT):
        print(periodDescription)
        print('MSE: ', mean_squared_error(y_test, predictedOT))
        print('MAE: ', mean_absolute_error(y_test, predictedOT))
        print('\n')