import pandas as pd
import numpy as np

class DataLoader(): 
    def fetch(self, filename, toDate, fromDate):
        data = pd.read_csv("{0}".format(filename))
        dataRange = data[(data['date'] <= '{0} 23:00:00'.format(toDate)) & (data['date'] >= '{0} 23:00:00'.format(fromDate))]
        return dataRange
    
    def filter(self, data):
        print("not implemented yet")