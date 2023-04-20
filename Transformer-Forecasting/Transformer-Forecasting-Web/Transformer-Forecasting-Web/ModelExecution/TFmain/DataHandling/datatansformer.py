import pandas

data = pandas.read_csv("DataHandling\ETTh1.csv")

data['datetime'] = pandas.to_datetime(data['date'])

columns = ['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp']    
data[columns] = data.apply(lambda row: pandas.Series({'year': row.datetime.year,"month":row.datetime.month, "day":row.datetime.day, "hour":row.datetime.hour, "weekday":row.datetime.weekday(), "weekofyear":row.datetime.weekofyear, "quarter":row.datetime.quarter, 'OilTemp': row.OT}), axis=1)

df = pandas.DataFrame(data, columns=['year', 'month', 'day', 'hour', 'weekday', 'weekofyear', 'quarter', 'OilTemp'])

data.to_csv('DataHandling\ETTh1-linear-format.csv', index=False)

