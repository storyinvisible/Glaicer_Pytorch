import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=[256, 128, 64, 32], p=0.5, use_last_year_smb=False, **args):
        super(LSTMPredictor, self).__init__()
        assert "bidirection" in args
        assert "n_layers" in args
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim, hidden_dim]
        if use_last_year_smb:
            input_dim = input_dim + 1
        self.use_last_year_smb = use_last_year_smb
        self.args = args
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim[0], batch_first=True,
                            bidirectional=self.args["bidirection"],
                            num_layers=self.args["n_layers"])
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=p)
        if self.args["bidirection"]:
            self.output = nn.Linear(hidden_dim[0] * 2, hidden_dim[1])
        else:
            self.output = nn.Linear(hidden_dim[0], hidden_dim[1])
        layer_reset = []
        for i in range(1, len(hidden_dim) - 1):
            layer_reset.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layer_reset.append(nn.Tanh())
        layer_reset.append(nn.Linear(hidden_dim[-1], 1))
        self.layer_final = nn.Sequential(*layer_reset)

    def forward(self, x, last_dmdt=None):
        if self.use_last_year_smb:
            assert last_dmdt is not None
            assert last_dmdt.shape == (x.shape[0], 1)
            x = torch.cat([x, last_dmdt], dim=1)
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        out = self.output(x)
        out = self.layer_final(out)
        return out


class LSTMPredictor3D(nn.Module):
    def __init__(self, input_dim=224, hidden_dim=[224, 128, 64, 32], p=0.5, use_last_year_smb=False,**args):
        super(LSTMPredictor, self).__init__()
        assert "bidirection" in args
        assert "n_layers" in args
        self.args = args
        if use_last_year_smb:
            input_dim = input_dim + 1
        self.use_last_year_smb = use_last_year_smb
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim[0], batch_first=True,
                            bidirectional=self.args["bidirection"],
                            num_layers=self.args["n_layers"])
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=p)
        if self.args["bidirection"]:
            self.output = nn.Linear(hidden_dim[0] * 2, hidden_dim[1])
        else:
            self.output = nn.Linear(hidden_dim[0], hidden_dim[1])
        layer_reset = []
        for i in range(1, len(hidden_dim) - 1):
            layer_reset.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layer_reset.append(nn.Tanh())
        layer_reset.append(nn.Linear(hidden_dim[-1], 1))
        self.layer_final = nn.Sequential(*layer_reset)

    def forward(self, x, last_dmdt=None):
        if self.use_last_year_smb:
            assert last_dmdt is not None
            assert last_dmdt.shape == (x.shape[0], 1)
            x = torch.cat([x, last_dmdt], dim=1)
        x = torch.unsqueeze(x, 1)
        x, _ = self.lstm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        out = self.output(x)
        out = self.layer_final(out)
        return out
