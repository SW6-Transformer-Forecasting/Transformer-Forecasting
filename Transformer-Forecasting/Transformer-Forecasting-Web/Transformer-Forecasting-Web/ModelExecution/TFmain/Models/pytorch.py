import torch
import pandas as pd
import numpy as np
from torch import nn

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import dataFilter as dl
import normalize_data as normalizer


macOS = False

periodDescription = sys.argv[1]
startPredictDate = sys.argv[2]
endPredictDate = sys.argv[3]
new_data = sys.argv[4]

cwd = os.getcwd()

if (new_data == "True"):
    dfilter = dl.DataFilter()
    data_to_filter = dfilter.fetch(cwd + '\ModelExecution\TFmain\Data\ETTh1.csv' if macOS == False else cwd + '/ModelExecution/TFmain/Data/ETTh1.csv', startPredictDate, endPredictDate)
    dfilter.execute(data_to_filter, cwd)

data = pd.read_csv(cwd + "\ModelExecution\TFmain\Data\cleandata.csv" if macOS == False else cwd + "/ModelExecution/TFmain/Data/cleandata.csv")
norm = normalizer.NormalizedData()

X = norm.normalize_data(data)
Y = data['OT']

x = torch.tensor(X, dtype=torch.float32)
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
load_model = True
if (load_model == True):
    model.load_state_dict(torch.load(cwd + "/ModelExecution/TFmain/Models/MSE_Y.pth"))
model.eval()

def train_model():
    # L1Loss = MAE
    # MSELoss = MSE (We focus here)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

    n_epochs = 3
    batch_size = 1

    for epoch in range(n_epochs):
        count = 0
        loss_amount = 0
        for i in range(0, len(x), batch_size):
            Xbatch = x[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            if (loss <= 3):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_amount += loss
                count += 1
        print(f'Finished epoch {epoch} - Est. Loss MSE: {loss_amount/count} - Count: {count}')

    torch.save(model.state_dict(), cwd + "/ModelExecution/TFmain/Models/MSE_Y.pth")
    print("Saved PyTorch Model State")

def predict_future():
    predictions = model(x[0:8])
    return predictions

train_model()
print(predict_future())