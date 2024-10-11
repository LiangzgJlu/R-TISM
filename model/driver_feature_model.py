import torch
import torch.nn as nn

class DriverFeatureModel(nn.Module):
    def __init__(self, feature, hidden_size, num_layers, *args, **kwargs) -> None:
        super(DriverFeatureModel, self).__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.feature = feature
        self.num_layers = num_layers
        self.lstm = nn.LSTM(feature, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) 
   
    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(x.device)

        output, (h, c) = self.lstm(x, (h_0, c_0))
        out = self.fc(output[:, -1, :].squeeze(1))
        return out
    
