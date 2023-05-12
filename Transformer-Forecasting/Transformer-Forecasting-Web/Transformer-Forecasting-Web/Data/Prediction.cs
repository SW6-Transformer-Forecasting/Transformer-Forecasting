namespace Transformer_Forecasting_Web.Data
{
    public class Prediction
    {
        public Prediction(int id, string model, string periodDescription)
        {
            Id = id;
            Model = model;
            PeriodDescription = periodDescription;
        }

        public Prediction() { }
        public int Id { get; }
        public string Model { get; }
        public string PeriodDescription { get; }
    }
}