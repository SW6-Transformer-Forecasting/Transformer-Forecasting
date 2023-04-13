using System;
using System.Diagnostics;

namespace Transformer_Forecasting_Web.ModelExecution
{
    public class ModelExecutor
    {     
        public void RunLinearModel(string periodDescription, string startPredictDate, string endPredictDate)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = @"C:\Users\krist\AppData\Local\Programs\Python\Python310\python.exe";
            
            string executionProductionPath = @"./ModelExecution/TFmain/executionproduction.py";
            start.Arguments = string.Format("{0} \"{1}\" \"{2}\" \"{3}\"", executionProductionPath, periodDescription, startPredictDate, endPredictDate);
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;

            string errors, results;
            using (Process process = Process.Start(start))
            {
                errors = process.StandardError.ReadToEnd();
                results = process.StandardOutput.ReadToEnd();
            }
            Console.WriteLine("Errors: " + errors);
            Console.WriteLine("Results: " + results);
        }

    }
}
