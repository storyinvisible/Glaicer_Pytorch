import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, layers=None, input_dim=256, hidden_dim=256, n_layers=1, bi_direction=False, p=0.5):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=bi_direction,
                            num_layers=n_layers, batch_first=True)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=p)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        return self.output(x)
