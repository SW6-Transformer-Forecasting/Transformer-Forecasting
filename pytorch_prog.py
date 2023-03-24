import torch
import pandas as pd
import numpy as np
from torch import nn

dataset = np.loadtxt('pytest2.csv', delimiter=',')

X = dataset[:,0:6]
Y = dataset[:,6]

x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)


class NeuralNetwork(nn.Module):
   def __init__(self):
       super().__init__()
       self.flatten = nn.Flatten()
       self.linear_relu_stack = nn.Sequential(
           nn.Linear(6, 9),
           nn.ReLU(),
           nn.Linear(9, 6),
           nn.ReLU(),
           nn.Linear(6, 1),
       )

   def forward(self, x):
       x = self.flatten(x)
       logits = self.linear_relu_stack(x)
       return logits

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
model.eval()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 1
 
for epoch in range(n_epochs):
    for i in range(0, len(x), batch_size):
        Xbatch = x[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
