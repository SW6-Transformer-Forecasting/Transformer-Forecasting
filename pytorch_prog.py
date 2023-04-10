import torch
import pandas as pd
import numpy as np
from torch import nn
import dataFilter as dl
import normalize_data as normalizer

BAGHLANI = False

NEW_DATA = False  # HAVE THIS SENT BY THE USER

if NEW_DATA is True:
    dfilter = dl.DataFilter()
    data_to_filter = dfilter.fetch('Data\ETTh1.csv' if BAGHLANI is False else 'Data/ETTh1.csv', '2017-01-01', '2018-01-01')
    dfilter.execute(data_to_filter)

data = pd.read_csv("Data\cleandata.csv" if BAGHLANI is False else "Data/cleandata.csv")
norm = normalizer.NormalizedData()

x = norm.normalize_data(data, 1)
Y = data['OT']

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(Y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(6, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
LOAD_MODEL = True
if LOAD_MODEL is True:
    model.load_state_dict(torch.load("MSE_Y.pth"))
model.eval()

def train_model():
    # L1Loss = MAE
    # MSELoss = MSE (We focus here)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

    n_epochs = 30
    batch_size = 1

    for epoch in range(n_epochs):
        count = 0
        loss_amount = 0
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_pred = model(x_batch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            if loss <= 3:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_amount += loss
                count += 1
        print(f'Finished epoch {epoch} - Est. Loss MSE: {loss_amount/count} - Count: {count}')

    torch.save(model.state_dict(), "MSE_Y.pth")
    print("Saved PyTorch Model State")

def predict_future():
    predictions = model(x[0:8])
    return predictions

train_model()
#print(predict_future())