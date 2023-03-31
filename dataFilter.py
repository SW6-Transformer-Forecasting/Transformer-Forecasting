import torch
import pandas as pd
import numpy as np

class DataFilter(): 
    def fetch(self, filename, toDate, fromDate):
        data = pd.read_csv("{0}".format(filename))
        dataRange = data[(data['date'] <= '{0} 23:00:00'.format(toDate)) & (data['date'] >= '{0} 23:00:00'.format(fromDate))]
        sortedData = dataRange[['HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT']]
        return sortedData
    
    def filter(self, data):
        data = data.to_numpy()
        prevAxis = data[0]
        fr = True
        arr = []
        
        for ax in data: 
            counter = 0
            
            if (fr == True):
                arr = np.append(arr, ax)
                prevAxis = ax
            
            for item in ax:
                if (item == prevAxis[counter] and fr == False):
                    prevAxis = ax
                    break
                else:
                    if (item == 0.0):
                        prevAxis = ax
                        break
                    else:
                        arr = np.append(arr, item)
                
                counter += 1
                
            prevAxis = ax
            fr = False
            
        return arr

dloader = DataFilter()
data = dloader.fetch('ETTh1.csv', '2016-12-05', '2016-12-04')
#print(data)
filteredData = dloader.filter(data)
print(filteredData)
