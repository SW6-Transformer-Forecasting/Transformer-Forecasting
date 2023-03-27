import torch
import pandas as pd
import numpy as np
from torch import nn

dataset = np.loadtxt('pytest3.csv', delimiter=',')

X = dataset[:,0:6]
Y = dataset[:,6]

x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)


# dataset = pd.read_csv('pytest2.csv')

# tensortest = torch.tensor(dataset.to_numpy(), dtype=torch.float32)

# print(tensortest)


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
       )

   def forward(self, x):
       x = self.flatten(x)
       logits = self.linear_relu_stack(x)
       return logits

model = NeuralNetwork()
load = True
if (load == True):
    model.load_state_dict(torch.load("model3.pth"))
model.eval()


# L1Loss = MAE
# MSELoss = MSE
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

n_epochs = 100
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
            #print("{0} - {1} - {2} - {3}".format(Xbatch, y_pred, ybatch, loss))
            
    print(f'Finished epoch {epoch}, loss est. {loss_amount/count} - Count: {count}')

torch.save(model.state_dict(), "model3.pth")
print("Saved PyTorch Model State to model.pth")
