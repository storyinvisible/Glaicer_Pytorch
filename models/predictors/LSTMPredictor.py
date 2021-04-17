import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, layers=None, input_dim=256, hidden_dim=256, n_layers=1, bi_direction=False, p=0.5, device=None):
        super(LSTMPredictor, self).__init__()
        self.device = device if device is None else torch.device("cpu")
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=bi_direction,
                            num_layers=n_layers)
        if n_layers == 1:
            self.hidden = (torch.zeros((n_layers, 1, hidden_dim)).to(device),
                           torch.zeros((n_layers, 1, hidden_dim)).to(device))
            self.output = nn.Linear(in_features=hidden_dim, out_features=1)
        else:
            self.hidden = (torch.zeros((n_layers * 2, 1, hidden_dim)).to(device),
                           torch.zeros((n_layers * 2, 1, hidden_dim)).to(device))
            self.output = nn.Linear(in_features=hidden_dim * 2, out_features=1)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x, (hn, cn) = self.lstm(x, self.hidden)
        self.hidden = (hn, cn)
        x = self.activation(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        return self.output(x)
