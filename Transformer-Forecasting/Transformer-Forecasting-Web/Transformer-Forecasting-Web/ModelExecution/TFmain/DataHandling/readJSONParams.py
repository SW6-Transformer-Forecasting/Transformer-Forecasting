import json
import os

cwd = os.getcwd()

jsonFile = open(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\params.json")
# jsonFile = open(cwd + "\params.json")

data = json.load(jsonFile)

Year = data[0]["Item2"]
Month = data[1]["Item2"]
Day = data[2]["Item2"]
Hour = data[3]["Item2"]
Weekday = data[4]["Item2"]
Weekofyear = data[5]["Item2"]
Quarter = data[6]["Item2"]
HoursToPredict = data[7]["Item2"]
TrainingPeriodLength = data[8]["Item2"]

class JsonParams(): 
    HoursToPredict = data[7]["Item2"]
    TrainingPeriodLength = data[8]["Item2"]
    
    def GetIncludedDateValues():
        dateValues = []
        if(Year): dateValues.append("year")
        if(Month): dateValues.append("month")
        if(Day): dateValues.append("day")
        if(Hour): dateValues.append("hour")
        if(Weekday): dateValues.append("weekday")
        if(Weekofyear): dateValues.append("weekofyear")
        if(Quarter): dateValues.append("quarter")
        
        return dateValues
    
    def GetNotIncludedDateValues():
        dateValues = []
        if(not Year): dateValues.append("year")
        if(not Month): dateValues.append("month")
        if(not Day): dateValues.append("day")
        if(not Hour): dateValues.append("hour")
        if(not Weekday): dateValues.append("weekday")
        if(not Weekofyear): dateValues.append("weekofyear")
        if(not Quarter): dateValues.append("quarter")
        
        return dateValues   
    