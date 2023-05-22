from DataHandling.datahandlerproduction import DataHandler
from Models.linearProduction import LinearRegression
from Models.pytorch import PyTorch
from SQL.queryexecutor import QueryExecutor
from DataHandling.datatransformerproduction import TransformData
from DataHandling.datatransformerproduction import PytorchTransformer
from DataHandling.dataFilter import DataFilter
from DataHandling.readJSONParams import JsonParams
import numpy
import pandas
import torch
import sys
import math
import os

new_data = False
periodDescription = sys.argv[1]
# hoursToPredictAhead = sys.argv[2]

cwd = os.getcwd()

if (new_data == "True"):
        dfilter = DataFilter()
        data_to_filter = dfilter.fetch(cwd + '\ModelExecution\TFmain\Data\ETTh1.csv')
        dfilter.execute(data_to_filter, cwd, False)

data = pandas.read_csv(cwd + "\ModelExecution\TFmain\Data\cleandata.csv")

# Linear model
# This instance stores the MinMaxScaler for later use when the normalization has to be inversed
dataTransformer = TransformData()

dataHandler = DataHandler(periodDescription, dataTransformer)

# Uses the prepared data and creates Linear models for the prediction task
modelData = dataHandler.linearModelInformation

linearRegression = LinearRegression(0, modelData.x_train, modelData.y_train, modelData.x_predict)

predictedOTDataframe = pandas.DataFrame(linearRegression.predictedOT, columns=["OT"])

predictedOTInversed = dataTransformer.InverseOT(predictedOTDataframe)


# SQL Starts here
max_identifiers = QueryExecutor.SelectQuery("SELECT MAX(group_id), MAX(row_id) FROM group_predictions")
print(max_identifiers)
max_group_id = max_identifiers[0][0]
max_row_id = max_identifiers[0][1]

print(max_group_id)

# makes sure that a new prediction has a higher ID value than the previous prediction
if (max_group_id != None and max_row_id != None):
    max_group_id = int(max_group_id) + 1
    max_row_id = int(max_row_id) + 1
else:
    max_group_id = 1
    max_row_id = 1

x_predict = pandas.DataFrame(modelData.x_predict, columns=JsonParams.GetIncludedDateValues())
x_predict["OT"] = 0

x_predict = dataTransformer.InverseDates(x_predict)

dateStamp = ""
for x in range(linearRegression.predictedOT.size):
    # retrieves the date of the prediction in the format DD-MM HH:00
    day = math.trunc(x_predict["day"][x])
    month = math.trunc(x_predict["month"][x])
    hour = math.trunc(x_predict["hour"][x])
    
    dateStamp = f"{day}-{month} {hour}:00"
    
    QueryExecutor.InsertQuery("INSERT INTO group_predictions (group_id, row_id) VALUES (%s, %s)",
                              (max_group_id, max_row_id))
    
    QueryExecutor.InsertQuery("INSERT INTO linear_predictions (row_id, dateStamp, OTPrediction) VALUES (%s, %s, %s)",
                        (max_row_id, dateStamp, float(predictedOTInversed[x][0])))
    
    max_row_id += 1
    
# prediction_description is inserted outside of the loop because we only need to store one single prediction period for the ID
QueryExecutor.InsertQuery("INSERT INTO prediction_descriptions (group_id, description) VALUES(%s, %s)",
                            (max_group_id, periodDescription))    

print("Success")