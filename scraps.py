import torch
import pandas as pd
import numpy as np
from torch import nn

# # Current type: Dataframes
# train_data = pd.read_csv("pytrain.csv").drop(['OT', 'date'], axis=1)
# test_data = pd.read_csv("pytest.csv").drop(['OT', 'date'], axis=1)

# new_test = pd.read_csv("pytest.csv").drop(['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'], axis=1)

# # Tensors
# train_tensor = torch.from_numpy(train_data.to_numpy())
# test_tensor = torch.from_numpy(test_data.to_numpy())

# new_test_tensor = torch.from_numpy(new_test.to_numpy())

# # Get cpu or gpu device for training.
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using {device} device")

# # Define model
# class NeuralNetwork(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.flatten = nn.Flatten()
#        self.linear_relu_stack = nn.Sequential(
#            nn.Linear(6, 12),
#            nn.ReLU(),
#            nn.Linear(12, 6),
#            nn.ReLU(),
#            nn.Linear(6, 1),
#        )

#    def forward(self, x):
#        x = self.flatten(x)
#        logits = self.linear_relu_stack(x)
#        return logits