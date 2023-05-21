import os
import pandas
import numpy
import torch
from torch import nn
from Models.pytorch import PyTorch
from DataHandling.datatransformerproduction import TransformData
from sklearn.model_selection import train_test_split
from DataHandling.dataFilter import DataFilter

cwd = os.getcwd()
dataTransformer = TransformData()
new_data = False

if (new_data == True):
        dfilter = DataFilter()
        data_to_filter = dfilter.fetch("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\ETTh1.csv")
        dfilter.execute(data_to_filter, cwd)

data = pandas.read_csv("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\cleandata_test.csv")

# We ignore train set here, as we create it when in TEST_MODE in the model
train, test = train_test_split(data, test_size=0.1, shuffle=False)
discord, test_samples = train_test_split(test, test_size=0.01, shuffle=False)
OT_data = test_samples[['OT']]
OT_data = OT_data.to_numpy()

pytorch = PyTorch(cwd, data, dataTransformer, False, True) # Bools: load_model & TEST_MODE
pytorch.train_model(cwd, False, False)

loss_fn = nn.MSELoss()

scaler = dataTransformer.getScaler()

tensor_data = test_samples[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
tensor_values = dataTransformer.FitAndTransformData(tensor_data)

scaler_reset = dataTransformer.FitAndTransformData(OT_data) # Has no use besides changing the stored scaler shape

transformed_tensor = torch.tensor(tensor_values, dtype=torch.float32)

predictions = numpy.zeros(shape=(12, 1))
index = 0

for x in transformed_tensor:
    x = x.unsqueeze(0)
    predictions[index] = pytorch.predict_future(x)
    index += 1
    
inversed_prediction = scaler.inverse_transform(predictions)
print(inversed_prediction)

inversed_prediction = scaler.inverse_transform(predictions)

index = 0
for x in range(OT_data.size):
    print(f"{OT_data[index][0]} and {inversed_prediction[index][0]}")
    index +=1

from_data = torch.tensor(OT_data, dtype=torch.float32)
predicted_data = torch.tensor(inversed_prediction, dtype=torch.float32)

print('Loss in MSE: ', loss_fn(from_data, predicted_data))