using Dapper;
using MySql.Data.MySqlClient;
using System.Data;

namespace kristofferhusdata.DataLibrary
{
    public class DataAccess : IDataAccess
    {
        public async Task<List<T>> LoadData<T, U>(string query, U parameters, string connectionString)
        {
            using (IDbConnection connection = new MySqlConnection(connectionString))
            {
                var rows = await connection.QueryAsync<T>(query, parameters);

                return rows.ToList();
            }
        }
    }
}
