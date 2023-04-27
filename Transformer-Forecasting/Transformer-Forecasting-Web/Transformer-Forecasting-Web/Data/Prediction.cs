namespace Transformer_Forecasting_Web.Data
{
    public class Prediction
    {
        public Prediction(int id, string model, string periodDescription,
            DateTime? startDate, DateTime? endDate)
        {
            Id = id;
            Model = model;
            PeriodDescription = periodDescription;
            StartDate = startDate;
            EndDate = endDate;
        }

        public Prediction() { }

        public int Id { get; }
        public string Model { get; }
        public string PeriodDescription { get; }
        public DateTime? StartDate { get; }
        public TimeSpan? StartTime { get; }
        public DateTime? EndDate { get; }
        public TimeSpan? EndTime { get; }


    }
}