using System;
using System.Diagnostics;

namespace Transformer_Forecasting_Web.ModelExecution
{
    public class ModelExecutor
    {     
        public void RunLinearModel1(string periodDescription, string startPredictDate, string endPredictDate)
        {
            var psi = new ProcessStartInfo();
            psi.FileName = @"C:\Users\krist\AppData\Local\Programs\Python\Python310";


            string script = @"./Transformer-Forecasting-main/executionProduction.py";


            psi.Arguments = $"\"{script}\"\"{periodDescription}\"\"{startPredictDate}\"\"{endPredictDate}\"";

            psi.UseShellExecute = true;
            psi.CreateNoWindow = true;

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

        public void RunLinearModel(string periodDescription, string startPredictDate, string endPredictDate)
        {
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = @"C:\Users\krist\AppData\Local\Programs\Python\Python310\python.exe";
            
            // arg[0] = Path to your python script (example : "C:\\add_them.py")
            // arg[1] = first arguement taken from  C#'s main method's args variable (here i'm passing a number : 5)
            // arg[2] = second arguement taken from  C#'s main method's args variable ( here i'm passing a number : 6)
            // pass these to your Arguements property of your ProcessStartInfo instance
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
