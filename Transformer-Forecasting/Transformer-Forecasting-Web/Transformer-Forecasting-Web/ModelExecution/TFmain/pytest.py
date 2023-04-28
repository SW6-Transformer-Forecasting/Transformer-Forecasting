import os
import pandas
import numpy
import torch
from torch import nn
from Models.pytorch import PyTorch
from DataHandling.datatransformerproduction import TransformData
from sklearn.model_selection import train_test_split

cwd = os.getcwd()
dataTransformer = TransformData()

data = pandas.read_csv("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\cleandata.csv")

train, test = train_test_split(data, test_size=0.004, random_state=42)
OT_data = test[['OT']]
OT_data = OT_data.to_numpy()

pytorch = PyTorch(cwd, train, dataTransformer, False)
pytorch.train_model(cwd)
predictions = pytorch.predict_future(dataTransformer)

loss_fn = nn.MSELoss()

scaler = dataTransformer.getScaler()

inversedPrediction = scaler.inverse_transform(predictions)
print(inversedPrediction)

index = 0
for x in range(OT_data): #shit breaks here
    print(f"{OT_data[index][0]} and {predictions[index][0]}")
    index +=1

print('MSE: ', loss_fn(test, predictions))