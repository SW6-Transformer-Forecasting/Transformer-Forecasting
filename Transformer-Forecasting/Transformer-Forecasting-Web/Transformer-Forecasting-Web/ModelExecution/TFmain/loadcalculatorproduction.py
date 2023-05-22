from Models.pytorch import PyTorch
from DataHandling.datatransformerproduction import PytorchTransformer
from DataHandling.dataFilter import DataFilter
from SQL.queryexecutor import QueryExecutor
import numpy
import pandas
import torch
import sys
import os


object_tuple = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

array_of_values = numpy.zeros(shape=(6, 1))
index = 0
for item in object_tuple:
    val = item.strip().replace(',', '.')
    array_of_values[index] = float(val)
    index += 1
    

df = {'HUFL': [array_of_values[0]], 'HULL': [array_of_values[1]], 'MUFL': [array_of_values[2]],
             'MULL': [array_of_values[3]],'LUFL': [array_of_values[4]], 'LULL': [array_of_values[5]]}

input_of_loads = pandas.DataFrame(data=df)

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

transformed_data = pytorch_transformer.transform_loads(input_of_loads)
tensor_values = torch.tensor(transformed_data, dtype=torch.float32)
# tensor_values = tensor_values.unsqueeze(0) # In case of broadcast error, add this line

prediction = pytorch.predict_future(tensor_values)

inversed_prediction = pytorch_transformer.inverse_OT(prediction)


QueryExecutor.ResetCalculationsTable()

QueryExecutor.InsertQuery("INSERT INTO calculations VALUES(%s)", inversed_prediction[0])