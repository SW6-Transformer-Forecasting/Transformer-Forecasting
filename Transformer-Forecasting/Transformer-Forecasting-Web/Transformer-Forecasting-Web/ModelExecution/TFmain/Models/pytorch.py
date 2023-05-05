import torch
from torch import nn
from sklearn.model_selection import train_test_split

class PyTorch:

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
    x = 0
    y = 0
        
    def __init__(self, cwd, data, dataTransformer, load_model = False, TEST_MODE = False):
        if (TEST_MODE == True):
            # We ignore test set here, as we dont need that on the model in TEST_MODE, as its in the Test code
            train, test = train_test_split(data, test_size=0.004, shuffle=False)
            data = train
        self.setup_data(data, dataTransformer)
        if (load_model == True):
            self.load_model(self.model, cwd, TEST_MODE)
        self.model.eval()
        

    def setup_data(self, data, dataTransformer):
        load_data = data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']]
        OT_data = data[['OT']]
        
        transformed_load_data = dataTransformer.FitAndTransformData(load_data)
        transformed_OT_data = dataTransformer.FitAndTransformData(OT_data)
        
        X = transformed_load_data
        Y = transformed_OT_data

        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1)
        
    def load_model(self, model, cwd, TEST_MODE):
        if(TEST_MODE == True):
            model.load_state_dict(torch.load("Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Models\MSE.pth"))
        else:
            model.load_state_dict(torch.load(cwd + "/ModelExecution/TFmain/Models/MSE.pth"))

    def train_model(self, cwd, save_model = False, TEST_MODE = False):
        # L1Loss = MAE
        # MSELoss = MSE (We focus here)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005)

        n_epochs = 200
        batch_size = 1

        for epoch in range(n_epochs):
            for i in range(0, len(self.x), batch_size):
                Xbatch = self.x[i:i+batch_size]
                y_pred = self.model(Xbatch)
                ybatch = self.y[i:i+batch_size]
                loss = loss_fn(y_pred, ybatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Finished epoch {epoch} - Est. Loss MSE: {loss}')

        if(TEST_MODE == True):
            # When testing
            torch.save(self.model.state_dict(), "Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Models\MSE.pth")
        elif(save_model == True):
            # From server
            torch.save(self.model.state_dict(), cwd + "/ModelExecution/TFmain/Models/MSE.pth")
        print("Saved PyTorch Model State")

    def predict_future(self):
        predictions = self.model(self.x[0:24]).detach().cpu().numpy()
        return predictions
        