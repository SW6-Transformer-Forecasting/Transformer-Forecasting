namespace Transformer_Forecasting_Web.Data
{
    public class Prediction
    {
        public int row_id { get; }
        public int group_id { get; }
        public float OT_prediction { get; }
        public string dateValue { get; }
    }
}