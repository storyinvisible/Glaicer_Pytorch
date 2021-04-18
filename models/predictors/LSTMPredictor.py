import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, p=0.5, **args):
        super(LSTMPredictor, self).__init__()
        assert "bidirection" in args
        assert "n_layers" in args
        self.args = args
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True,
                            bidirectional=self.args["bidirection"],
                            num_layers=self.args["n_layers"])
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=p)
        if self.args["bidirection"]:
            self.output = nn.Linear(hidden_dim * 2, 1)
        else:
            self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        out = self.output(x)
        return out
