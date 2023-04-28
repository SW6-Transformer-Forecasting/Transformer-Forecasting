import mysql.connector

databaseConnection = mysql.connector.connect(
    host="transformer-data.mysql.database.azure.com",
    user="Gud",
    password="#patrickpatrickjuliankristoffer123",
    database="transformerpredictiondata"
)

cursor = databaseConnection.cursor()

class QueryExecutor:
    def InsertQuery(query, arguments):
        cursor.execute(query, arguments)
        databaseConnection.commit()
        
    def SelectQuery(query):
        cursor.execute(query)
        return cursor.fetchall()