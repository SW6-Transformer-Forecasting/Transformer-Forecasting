namespace Transformer_Forecasting_Web.SQL
{
    public interface IDataAccess
    {
        Task<List<T>> LoadData<T, U>(string query, U parameters, string connectionString);
    }
}