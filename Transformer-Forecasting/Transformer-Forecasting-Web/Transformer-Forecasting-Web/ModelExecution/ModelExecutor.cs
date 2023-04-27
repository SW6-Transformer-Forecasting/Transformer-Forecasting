using System;
using System.Diagnostics;
using System.Drawing.Printing;

namespace Transformer_Forecasting_Web.ModelExecution
{
    public class ModelExecutor
    {
        public void RunLinearModel(string periodDescription, string startPredictDate, string endPredictDate, int model = 1, bool newData = false)
        {
            ProcessStartInfo start = new ProcessStartInfo();

            // Windows
            var Local = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), @"Programs\Python"); // Gets Appdata/Local folder path
            start.FileName = Directory.GetFiles(Local, "python.exe", SearchOption.AllDirectories)[0]; // Searches for python.exe within the Local folder, no matter version, and returns first result

            // Mac
            // var macPyLib = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), @"/Library/Frameworks/Python.framework/Versions/");
            // start.FileName = Directory.GetFiles(macPyLib, "Python", SearchOption.AllDirectories)[0]; // Searches for python.exe within the weirdness of the MacOS, no matter version, and returns first result

            string executionProductionPath = @"./ModelExecution/TFmain/Models/pytorch.py"; //Make this some standard
            if (model == 1)
            {
                executionProductionPath = @"./ModelExecution/TFmain/Models/pytorch.py";
            }
            if(model == 2)
            {
                executionProductionPath = @"./ModelExecution/TFmain/executionproduction.py";
            }
            
            start.Arguments = string.Format("{0} \"{1}\" \"{2}\" \"{3}\" \"{4}\"", executionProductionPath, periodDescription, startPredictDate, endPredictDate, newData);
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
