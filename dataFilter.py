import torch
import pandas as pd
import numpy as np

class DataFilter(): 
    def fetch(self, filename, fromDate, toDate):
        data = pd.read_csv("{0}".format(filename))
        dataRange = data[(data['date'] <= '{0} 23:00:00'.format(toDate)) & (data['date'] >= '{0} 23:00:00'.format(fromDate))]
        sortedData = dataRange[['HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT']]
        return sortedData
    
    def filter(self, data):
        data = data.drop_duplicates(subset=['HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT'])
        data = data.to_numpy()
        
        appendFlag = True
        prevAxis = [0] * 7 # Init of array filled with 0's to avoid double line data
        arr = []
        
        for row in data: 
            counter = 0
            
            for item in row:
                if (item == prevAxis[counter]):
                    appendFlag = False
                    break
                else:
                    if (item == 0.0):
                        appendFlag = False
                        break
                    
                counter += 1
                
            if (appendFlag == True):
                arr = np.append(arr, row)
            
            prevAxis = row
            appendFlag = True
            
        return arr
    
    def saveFile(self, data):
        print("Not Implemented yet")

dloader = DataFilter()
data = dloader.fetch('ETTh1.csv', '2018-01-29', '2018-02-01')
#print(data)
filteredData = dloader.filter(data)
print(filteredData)