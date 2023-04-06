using System.Collections.Generic;
using System.Threading.Tasks;

namespace kristofferhusdata.DataLibrary
{
    public interface IDataAccess
    {
        Task<List<T>> LoadData<T, U>(string query, U parameters, string connectionString);
    }
}