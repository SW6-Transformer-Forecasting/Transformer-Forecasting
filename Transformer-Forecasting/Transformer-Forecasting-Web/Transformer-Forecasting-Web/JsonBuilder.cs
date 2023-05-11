using Newtonsoft;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class GlobalJsonBuilder
{
    public static readonly JsonBuilder jsonBuilder = new JsonBuilder();

    public class JsonBuilder
    {
        class Parameter
        {
            private string paramName;
            private bool paramStatus;

            public Parameter(string name, bool status)
            {
                paramName = name;
                paramStatus = status;
            }
        }
        Parameter CreateObject(string paramName, bool status)
        {
            Parameter param = new Parameter(paramName, status);
            return param;
        }

        List<Parameter> parameters = new List<Parameter>();

        public void AddParam(string paramName, bool status)
        {
            parameters.Add(CreateObject(paramName, status));
        }

        public void BuildJson()
        {
            parameters.Sort();
            var jsonSetting = JsonConvert.SerializeObject(parameters, Formatting.Indented);
            File.WriteAllText("params.json", jsonSetting);
        }
    }
}