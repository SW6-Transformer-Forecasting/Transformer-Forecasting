import torch
from torch import nn

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
        
    def __init__(self, cwd, data, dataTransformer, load_model):
        self.setup_data(data, dataTransformer)
        self.load_model(self.model, cwd, load_model)
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
        
    def load_model(self, model, cwd, load_model = False):
        if (load_model == True):
            model.load_state_dict(torch.load(cwd + "/ModelExecution/TFmain/Models/MSE.pth"))

    def train_model(self, cwd):
        # L1Loss = MAE
        # MSELoss = MSE (We focus here)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0005)

        n_epochs = 100
        batch_size = 1

        for epoch in range(n_epochs):
            count = 0
            loss_amount = 0
            for i in range(0, len(self.x), batch_size):
                Xbatch = self.x[i:i+batch_size]
                y_pred = self.model(Xbatch)
                ybatch = self.y[i:i+batch_size]
                loss = loss_fn(y_pred, ybatch)
                if (loss <= 3):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_amount += loss
                    count += 1
            print(f'Finished epoch {epoch} - Est. Loss MSE: {loss_amount/count} - Count: {count}')

        torch.save(self.model.state_dict(), cwd + "/ModelExecution/TFmain/Models/MSE.pth") # From server
        #torch.save(self.model.state_dict(), "Transformer-Forecasting\Transformer-Forecasting-Web\Transformer-Forecasting-Web\ModelExecution\TFmain\Models\MSE.pth") # When testing
        print("Saved PyTorch Model State")

    def predict_future(self, dataTransformer):
        predictions = self.model(self.x[0:24]).detach().cpu().numpy()
        return predictions
        predicted_OTs = []
        for item in range(predictions):
            predicted_OTs.insert(dataTransformer.InverseOT(item, 6, True))
        