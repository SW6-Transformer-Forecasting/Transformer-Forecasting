from DataHandling.datahandlerproduction import DataHandler
from Models.linearProduction import LinearRegression
from SQL.queryexecutor import QueryExecutor
import sys

periodDescription = sys.argv[1]
startPredictDate = sys.argv[2]
endPredictDate = sys.argv[3]

# Picks the training/test periods specified in TrainTestPeriods.json and prepares data for Linear model training
dataHandler = DataHandler(periodDescription, startPredictDate, endPredictDate)

# Uses the prepared data and creates Linear models for the prediction task
modelData = dataHandler.linearModelInformation

linearRegression = LinearRegression(modelData.periodDescription, modelData.x_train, modelData.y_train, modelData.x_predict)

max_identifiers = QueryExecutor.SelectQuery("SELECT MAX(group_id), MAX(row_id) FROM group_predictions")

max_group_id = max_identifiers[0][0]
max_row_id = max_identifiers[0][1]

# makes sure that a new prediction has a higher ID value than the previous prediction
if (max_group_id != None and max_row_id != None):
    max_group_id = int(max_group_id) + 1
    max_row_id = int(max_row_id) + 1
else:
    max_group_id = 1
    max_row_id = 1

# prediction_description is inserted outside of the loop because we only need to store one single prediction period for the ID
QueryExecutor.InsertQuery("INSERT INTO prediction_descriptions (group_id, description) VALUES(%s, %s)",
                            (max_group_id, periodDescription))

dateStamp = ""
for x in range(linearRegression.predictedOT.size):
    # retrieves the date of the prediction in the format DD-MM HH:00
    dateStamp = f"{modelData.x_predict['day'][x]}-{modelData.x_predict['month'][x]} {modelData.x_predict['hour'][x]}:00"
    
    QueryExecutor.InsertQuery("INSERT INTO group_predictions (group_id, row_id) VALUES (%s, %s)",
                              (max_group_id, max_row_id))
    
    QueryExecutor.InsertQuery("INSERT INTO linear_predictions (row_id, dateStamp, OTPrediction) VALUES (%s, %s, %s)",
                        (max_row_id, dateStamp, linearRegression.predictedOT[x][0]))
    
    max_row_id += 1
    
    
print("Success")