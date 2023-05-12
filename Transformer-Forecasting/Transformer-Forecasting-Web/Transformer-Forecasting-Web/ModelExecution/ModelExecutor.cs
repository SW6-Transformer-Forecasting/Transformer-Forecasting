using System;
using System.Diagnostics;
using System.Drawing.Printing;

namespace Transformer_Forecasting_Web.ModelExecution
{
    public class ModelExecutor
    {
        public void RunLinearModel(bool newData = false)
        {
            ProcessStartInfo start = new ProcessStartInfo();

            // Windows
            var Local = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Python"); // Gets Appdata/Local folder path
            start.FileName = Directory.GetFiles(Local, "python.exe", SearchOption.AllDirectories)[0]; // Searches for python.exe within the Local folder, no matter version, and returns first result

            string executionProductionPath = @"./ModelExecution/TFmain/executionproduction.py";

            // Json Builder
            GlobalJsonBuilder.jsonBuilder.BuildJson();
            GlobalJsonBuilder.jsonBuilder.ResetParam();

            start.Arguments = string.Format("{0} \"{1}\"", executionProductionPath, newData);
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
