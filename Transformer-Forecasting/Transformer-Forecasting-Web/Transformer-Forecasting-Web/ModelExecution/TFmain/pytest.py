import os
import pandas
import numpy
import torch
from torch import nn
from Models.pytorch import PyTorch
from DataHandling.datatransformerproduction import PytorchTransformer
from sklearn.model_selection import train_test_split
from DataHandling.dataFilter import DataFilter

cwd = os.getcwd()
pytorch_transformer = PytorchTransformer()
new_data = False

if (new_data == True):
        dfilter = DataFilter()
        data_to_filter = dfilter.fetch("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\ETTh1.csv")
        dfilter.execute(data_to_filter, cwd, True)

data = pandas.read_csv("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\cleandata.csv")

# We ignore train set here, as we create it when in TEST_MODE in the model
train, test = train_test_split(data, test_size=0.1, shuffle=False)

discard, test_samples = train_test_split(test, test_size=0.01, random_state=36765)
OT_data = test_samples[['OT']]
OT_data = OT_data.to_numpy()

pytorch = PyTorch(cwd, data, pytorch_transformer, True, True) # Bools: load_model & TEST_MODE

# MODEL IS PRE-TRAINED - WE DONT TRAIN NO MORE
# pytorch.train_model(cwd, True, True) # Bools: save_model & TEST_MODE

loss_fn = nn.MSELoss()

tensor_data = test_samples[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
tensor_values = pytorch_transformer.transform_loads(tensor_data)

transformed_tensor = torch.tensor(tensor_values, dtype=torch.float32)

predictions = numpy.zeros(shape=(12, 1))
index = 0

for x in transformed_tensor:
    x = x.unsqueeze(0)
    predictions[index] = pytorch.predict_future(x)
    index += 1

inversed_prediction = pytorch_transformer.inverse_OT(predictions)

from_data = torch.tensor(OT_data, dtype=torch.float32)
predicted_data = torch.tensor(inversed_prediction, dtype=torch.float32)

MSE_val = loss_fn(from_data, predicted_data)

print('Loss in MSE: ', MSE_val)