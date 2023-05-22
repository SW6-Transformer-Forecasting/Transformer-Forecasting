using Newtonsoft;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class GlobalJsonBuilder
{
    public static readonly JsonBuilder jsonBuilder = new JsonBuilder();

    public class JsonBuilder
    {
        List<Object> parameters = new List<Object>();

        public void AddParam(object obj)
        {
            parameters.Add(obj);
        }

        public void ResetParam()
        {
            parameters.Clear();
        }

        public void BuildJson()
        {
            //parameters.Sort();
            var jsonSetting = JsonConvert.SerializeObject(parameters, Formatting.Indented);
            File.WriteAllText("params.json", jsonSetting);
        }

        public List<Object> ReadJson()
        {
            var itemsFromJson = File.ReadAllText("params.json");
            return JsonConvert.DeserializeObject<List<Object>>(itemsFromJson);
        }
    }
}