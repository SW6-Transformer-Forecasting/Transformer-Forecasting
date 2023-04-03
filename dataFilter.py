import torch
import pandas as pd
import numpy as np

class DataFilter(): 
    def fetch(self, filename, fromDate, toDate):
        data = pd.read_csv("{0}".format(filename))
        dataRange = data[(data['date'] <= '{0} 23:00:00'.format(toDate)) & (data['date'] >= '{0} 23:00:00'.format(fromDate))]
        sortedData = dataRange[['HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT']]
        return sortedData
    
    def filterData(self, data):
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
            
            
        arr = np.split(arr, len(arr)/7)
        return arr
    
    def saveFile(self, data):
        df = pd.DataFrame(data, columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
        df.to_csv('cleandata.csv', index=False)

    def execute(self, data):
        dfilter = DataFilter()
        dfilter.saveFile(dfilter.filterData(data))