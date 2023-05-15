﻿namespace Transformer_Forecasting_Web.Data
{
    public class Prediction
    {
        public Prediction(int id, string model, string periodDescription, double[]? dTP_Value, double[]? lFP_Value)
        {
            Id = id;
            Model = model;
            PeriodDescription = periodDescription;
            DTP_Value = dTP_Value;
            LFP_Value = lFP_Value;
        }

        public Prediction() { }

        public Prediction(Prediction prediction) 
        { 
            this.Id = prediction.Id;
            this.Model = prediction.Model;
            this.PeriodDescription = prediction.PeriodDescription;
            this.DTP_Value = prediction.DTP_Value;
            this.LFP_Value = prediction.LFP_Value;
        }

        public int Id { get; }
        public string Model { get; }
        public string PeriodDescription { get; }
        public double[]? DTP_Value { get; }
        public double[]? LFP_Value { get; }

    }
}