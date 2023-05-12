
import json
import os

cwd = os.getcwd()


f = open(r"C:\Users\krist\source\repos\CAOS calc\Transformer-Forecasting\Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\params.json")

data = json.load(f)

for i in range(7, 16):
    print(data[i]['Item2'])




