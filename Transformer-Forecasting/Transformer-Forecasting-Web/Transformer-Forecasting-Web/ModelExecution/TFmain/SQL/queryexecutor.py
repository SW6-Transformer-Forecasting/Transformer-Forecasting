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
        print("Attempting to insert...")
        cursor.execute(query, arguments)
        print("Executed cursor...")
        databaseConnection.commit()
        print("Success!")
        
    def SelectQuery(query):
        print("Attempting to select...")
        cursor.execute(query)
        print("Success!")
        return cursor.fetchall()