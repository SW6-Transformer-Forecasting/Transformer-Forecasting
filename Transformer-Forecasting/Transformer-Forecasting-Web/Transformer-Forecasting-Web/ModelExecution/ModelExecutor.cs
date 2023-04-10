using System;
using System.Diagnostics;

namespace Transformer_Forecasting_Web.ModelExecution
{
    public class ModelExecutor
    {     
        void RunLinearModel()
        {
            var psi = new ProcessStartInfo();
            psi.FileName = @"C:\Users\krist\AppData\Local\Programs\Python\Python310";


            string script = @"./Transformer-Forecasting-main/execution.py";
            string startPredictDate = "";
            string endPredictDate = "";

            psi.Arguments = $"\"{script}\"\"{startPredictDate}\"\"{endPredictDate}\"";

            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardInput = true;
            psi.RedirectStandardError = true;

            string errors = "";
            string results = "";

            using(var process  = Process.Start(psi))
            {
                errors = process.StandardError.ReadToEnd();
                results = process.StandardOutput.ReadToEnd();
            }

            Console.Write($"Errors: \n  {errors} \n");
            Console.WriteLine($"Results: \n {results}");
        }
    }
}
