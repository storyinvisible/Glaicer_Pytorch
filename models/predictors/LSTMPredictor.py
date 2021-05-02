import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=[256, 128, 64, 32], p=0.5, **args):
        super(LSTMPredictor, self).__init__()
        assert "bidirection" in args
        assert "n_layers" in args
        self.args = args
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim[0], batch_first=True,
                            bidirectional=self.args["bidirection"],
                            num_layers=self.args["n_layers"])
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=p)
        if self.args["bidirection"]:
            self.output = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim//2),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim//2, 1))

        else:
            self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                        nn.Tanh(),
                                        nn.Linear(hidden_dim//2, 1))


    def forward(self, x):
        # print("input", x.shape)
        x = torch.unsqueeze(x, 1)
        # print("unsqueeze", x.shape)
        x, _ = self.lstm(x)
        # print("After LSTM")
        x = self.activation(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        # print("Flatten", x.shape)
        out = self.output(x)
        # print(out.shape)
        out = self.layer_final(out)
        # print(out.shape)
        return out
