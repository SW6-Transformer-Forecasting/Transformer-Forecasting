from datetime import datetime, timedelta


examplestring = "2018-12-01 13:00:00"


d = str(datetime.strptime(examplestring, "%Y-%m-%d %H:%M:%S") - timedelta(days=365))

# 2016-07-01 00:00:00

if(d < "2016-07-01 00:00:00"):
    d = "2016-07-01 00:00:00"

print(d)