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
        data = data.drop_duplicates(subset=['HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT'])
        data = data.to_numpy()
        prevAxis = data[0]
        firstRun = True
        arr = []
        
        for row in data: 
            counter = 0
            
            if (firstRun == True):
                arr = np.append(arr, row)
                prevAxis = row
            
            for item in row:
                if (item == prevAxis[counter] and firstRun == False):
                    prevAxis = row
                    break
                else:
                    if (item == 0.0):
                        prevAxis = row
                        break
                    else:
                        arr = np.append(arr, item) #Add some memory, so it ignores the entire axis if it contains a 0 value
                
                counter += 1
                
            prevAxis = row
            firstRun = False
            
        return arr

dloader = DataFilter()
data = dloader.fetch('ETTh1.csv', '2016-12-05', '2016-12-04')
#print(data)
filteredData = dloader.filter(data)
print(filteredData)
