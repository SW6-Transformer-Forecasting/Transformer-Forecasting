# 2018-06-26 19:00:00
import numpy
from datetime import datetime, timedelta


datething = "2018-06-26 19:00:00"
datetimething = datetime.strptime(datething, "%Y-%m-%d %H:%M:%S")
dates = []
for i in range(24):
    datetimething += timedelta(hours=1)
    dates += [str(datetimething)]


print(dates)
 