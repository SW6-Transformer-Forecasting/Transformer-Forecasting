import pandas 
import numpy 
from scipy import stats

class DataFilter(): 
    #, fromDate, toDate
    def fetch(self, filename):
        data = pandas.read_csv("{0}".format(filename))
        sortedData = data[['date', 'HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT']]
        return sortedData
    
    def filter_data(self, data):
        data = data.drop_duplicates(subset=['HUFL', 'HULL', 'MUFL', 'MULL','LUFL', 'LULL', 'OT'])
        data = data.to_numpy()
        
        appendFlag = True
        prevAxis = [0] * 8 # Init of array filled with 0's to avoid double line data. Multiplier is amount of items in each row
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
                arr = numpy.append(arr, row)
            
            prevAxis = row
            appendFlag = True
            
            
        arr = numpy.split(arr, len(arr)/8)
        return arr
    
    def saveFile(self, data, cwd, TEST_MODE):
        df = pandas.DataFrame(data, columns=['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
        if(TEST_MODE == True):
            df.to_csv("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Data\cleandata.csv", index=False) # For Local Test
        else:
            df.to_csv(cwd + "\ModelExecution\TFmain\Data\cleandata.csv", index=False) # For Production

    def execute(self, data, cwd, TEST_MODE):
        dfilter = DataFilter()
        dfilter.saveFile(dfilter.filter_data(data), cwd, TEST_MODE)
        
    