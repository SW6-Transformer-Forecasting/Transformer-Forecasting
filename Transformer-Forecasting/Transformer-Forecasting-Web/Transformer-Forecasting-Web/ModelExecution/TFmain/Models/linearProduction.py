from sklearn import linear_model

class LinearRegression:
    predictedOT = []
    
    def __init__(self, periodDescription, x_train, y_train, x_predict):
        linearModel = self.TrainLinearModel(x_train, y_train)
        self.predictedOT = linearModel.predict(x_predict)


    def TrainLinearModel(self, x_train, y_train):
        linearModel = linear_model.LinearRegression()
        return linearModel.fit(x_train, y_train)
    