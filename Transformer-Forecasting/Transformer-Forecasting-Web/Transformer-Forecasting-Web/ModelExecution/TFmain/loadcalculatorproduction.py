from Models.pytorch import PyTorch
from DataHandling.datatransformerproduction import PytorchTransformer
from DataHandling.dataFilter import DataFilter
import numpy
import pandas
import torch
import os

input_of_loads = numpy.zeros(shape=(1, 1)) # Some loads from front-end goes here instead

cwd = os.getcwd()

new_data = False
if (new_data == True):
        dfilter = DataFilter()
        data_to_filter = dfilter.fetch(cwd + '\ModelExecution\TFmain\Data\ETTh1.csv')
        dfilter.execute(data_to_filter, cwd, False)

data = pandas.read_csv(cwd + "\ModelExecution\TFmain\Data\cleandata.csv")

# Pytorch
pytorch_transformer = PytorchTransformer()

pytorch = PyTorch(cwd, data, pytorch_transformer, True)

train_model = False
if(train_model == True):
    pytorch.train_model(cwd, False) # Bool: save_model

# Remove this, add directly into a tensor further up (Mayb we cant do this, it needs to be normalized)
tensor_data = input_of_loads[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]

transformed_tensor = pytorch_transformer.transform_loads(tensor_data)
tensor_values = torch.tensor(transformed_tensor, dtype=torch.float32)
# tensor_values = tensor_values.unsqueeze(0) # In case of broadcast error, add this line

prediction = pytorch.predict_future(tensor_values)

inversed_prediction = pytorch_transformer.inverse_OT(prediction)

# Return results to front end here