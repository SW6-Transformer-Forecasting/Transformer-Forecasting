import mysql.connector

databaseConnection = mysql.connector.connect(
    host="transformerforecastingdata.mysql.database.azure.com",
    user="gud",
    password="transformereerseje2023!",
    database="data"
)

cursor = databaseConnection.cursor()

class QueryExecutor:
    def InsertQuery(query, arguments):
        cursor.execute(query, arguments)
        databaseConnection.commit()
        
    def SelectQuery(query):
        cursor.execute(query)
        return cursor.fetchall()