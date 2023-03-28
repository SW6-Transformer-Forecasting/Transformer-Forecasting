import pandas as pd
import csv

fields = ['date', 'OT']
data = pd.read_csv("ETTh1.csv", skipinitialspace=True, usecols=fields)

trainPeriod = data[(data['date'] <= '2018-01-08 00:00:00') & (data['date'] >= '2018-01-01 00:00:00')]
testPeriod = data[(data['date'] >= '2018-01-08 00:00:00') & (data['date'] <= '2018-01-09 00:00:00')]

trainFile = open('Generated Datasets/trainPeriod.csv', 'w', newline='')
testFile = open('Generated Datasets/testPeriod.csv', 'w', newline='')
trainWrite = csv.writer(trainFile)
testWrite = csv.writer(testFile)

header = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
          '20', '21', '22', '23']

trainWrite.writerow(header)

rowText = ''
for row in trainPeriod.itertuples(index=False):
    dateValue = row.date
    OTValue = str(row.OT)

    if dateValue[11:13] == '00' and rowText != '':
        trainWrite.writerow([rowText])
        rowText = OTValue
    else:
        rowText += OTValue

testWrite.writerow(header)

rowText = ''
for row in testPeriod.itertuples(index=False):
    dateValue = row.date
    OTValue = str(row.OT)

    if dateValue[11:13] == '00' and rowText != '':
        testWrite.writerow([rowText])
        rowText = OTValue
    else:
        rowText += OTValue



