# Time-Series Forecasting for Predicting the Oil Temperature in Electrical Transformers
The oil temperature of an electrical transformer has a big impact on the longevity of the transformer. This is due to
the insulation breakdown which happens at higher temperatures. In order for the operators of a transformer to
prepare for days where the oil temperature is too high, they would need to have a times-series machine learning
model that predicts ahead in time.

We use a dataset called ETTh1, which contains values about the date, electric load values and oil temperature for a
given transformer in China. With this dataset, it is possible to split the dates into different features, such as month,
day, hour and quarter. These features can then be used to perform multivariate time series forecasting.

We have used Linear Regression with sklearn to perform multivariate time series forecasting. We have through
research found out that linear regression performs well on the ETTh1 data. Before actually training the linear
regression model with the data, we perform some preprocessing steps. The first step is removing duplicate values
and 0-values from the dataset, as these values dont represent the actual data. The second step is performing outlier
detection. We remove outliers by training the model on all of the filtered data, followed by predicting on every single
value in the dataset with the trained model. We calculate the MAPE (Mean absolute percentage error) for every
prediction, and filter out the rows that have an error greater than 60%. This model is called the DTP model.

Experiments on this model has shown, that it can make predictions with an average of a 15% error. We concluded
that this error is acceptable, since the majority of the errors were between 6-11%. The reason why the error went up
to a 15% average is because the model gets 3 months of training context, which is not optimal. The reason why
the model is not trained on all of the data, is because the distribution shift in the ETTh1 dataset has a big negative
effect on the predictions. In order to tackle this, we will have to expand the DTP model with inspiration of NLinear,
which is a special form of normalization for linear regression.

We have also designed a load calculator, which uses a multi-layer perceptron to predict a single oil temperature
value from a set of electric load values. This model is built with PyTorch, and we call it the LFP model.

Experiments on the LFP model has shown that it can create predictions with a variable accuracy, but with good
data is can reach an average accuracy with an MSE of 1-2. We concluded that despite that the predictions could
reach an MSE of 22, it was highly situational and only reached these values when the data had very little coherence
with its values in the samples (outliers essentially). Another reason for its variable accuracy, is due to the model
being trained on the full dataset, after the worst outlier data is removed by the data filter, which gives it a lot of
smaller outliers that can skew its accuracy as they’re still present. However, it is currently necessary for the model
to get the full dataset, to create a better general linear function, which it uses for the predictions.

We have designed a webpage with Blazor Server, which makes it much easier for the transformer operator to
perform predictions and visualize said predictions. The webpage consists of a settings page that allows the user to
change most of the hyperparameters for the DTP model. The main page is an interactive table, which displays the
individual predictions on each row. Clicking on the row shows a dialog, which includes a graphical representation
of the predictions. There is also a page for the load prediction calculator, which allows the user to input 6 load
values to get an estimated oil temperature.

The communication between the Blazor front-end and the Python back-end is established through a MySQL database
and a JSON file. With this database, we can store the predictions of the DTP models permanently, and we can
temporarily store the predicted value from the LFP model. The JSON file acts as a way to save the settings of the
DTP model so that the Python back-end can use the saved settings to make predictions for the DTP model.
iii
