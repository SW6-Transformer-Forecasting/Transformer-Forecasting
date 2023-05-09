using Newtonsoft;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class JsonBuilder
{
    List<Object> parameters = new List<Object>();

    void AddParam(Object param)
    {
        parameters.Add(param);
    }

    void BuildJson()
    {
        var jsonSetting = JsonConvert.SerializeObject(parameters, Formatting.Indented);
        File.WriteAllText("params.json", jsonSetting);
    }
}
