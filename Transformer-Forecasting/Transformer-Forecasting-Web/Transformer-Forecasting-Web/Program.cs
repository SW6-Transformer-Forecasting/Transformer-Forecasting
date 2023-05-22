using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;

using Transformer_Forecasting_Web.Data;
using Transformer_Forecasting_Web.SQL;
using MudBlazor.Services;
using Transformer_Forecasting_Web.ModelExecution;


var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();
builder.Services.AddSingleton<IDataAccess, DataAccess>();
builder.Services.AddSingleton<DataAccess>();
builder.Services.AddSingleton<ModelExecutor>();
builder.Services.AddMudServices();



var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");

app.Run();
