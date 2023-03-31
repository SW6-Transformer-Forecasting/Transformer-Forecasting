import torch
import pandas as pd
import numpy as np
from torch import nn
import data_loader as dl

dloader = dl.DataFilter()
data = dloader.fetch('ETTh1.csv', '2018-01-07', '2018-01-01')
filteredData = dloader.filter(data)

X = data[['HUFL', 'HULL', 'MUFL','MULL','LUFL', 'LULL']]
Y = data['OT']

x = torch.tensor(X.to_numpy(), dtype=torch.float32)
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
           nn.ReLU(),
       )

   def forward(self, x):
       x = self.flatten(x)
       logits = self.linear_relu_stack(x)
       return logits

model = NeuralNetwork()
load = True
if (load == True):
    model.load_state_dict(torch.load("MSE.pth"))
model.eval()

# L1Loss = MAE
# MSELoss = MSE (We focus here)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

n_epochs = 1
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

#torch.save(model.state_dict(), "MSE.pth")
print("Saved PyTorch Model State")
