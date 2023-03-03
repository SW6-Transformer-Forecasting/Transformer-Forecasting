# This program takes the average electrical load of the high load values and plots it with the corresponding oil temperature
# The program helps give a visual understanding of how the electrical load and oil temperature affect each other

from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn import preprocessing


dataframe = pd.read_csv("ETTh1.csv")

arr = np.array([dataframe["HUFL"], dataframe["HULL"], dataframe["MUFL"], dataframe["MULL"], dataframe["LUFL"], dataframe["LULL"]])
arrMean = np.mean(arr)

HUFL = np.array(dataframe["HUFL"])
HULL = np.array(dataframe["HULL"])
MUFL = np.array(dataframe["MUFL"])
MULL = np.array(dataframe["MULL"])
LUFL = np.array(dataframe["LUFL"])
LULL = np.array(dataframe["LULL"])
OT = np.array(dataframe["OT"])

datapointsRange = 500
startPoint = 2000

meanLoad = []
oilTemp = [0 for i in range(datapointsRange)]
meanLoad = [0 for i in range(datapointsRange)]
for x in range(datapointsRange):
    #meanLoad[x] = (HUFL[x+startPoint] + HULL[x+startPoint] + MUFL[x+startPoint] + MULL[x+startPoint] + LUFL[x+startPoint] + LULL[x+startPoint]) / 6
    meanLoad[x] = (HUFL[x+startPoint] + MUFL[x+startPoint] + LUFL[x+startPoint]) / 3
    oilTemp[x] = OT[x+startPoint]


data2 = {'Load':meanLoad, 'Oil':oilTemp}



df = pd.DataFrame(data2)


plt.plot(df)

plt.show()
