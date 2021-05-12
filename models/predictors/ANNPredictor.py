import torch.nn as nn
import torch

class ANNPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, p=0.5, **args):
        super(ANNPredictor, self).__init__()
        self.args = args
        self.predictor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=p),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

    def forward(self, x):
        return self.predictor(x)


class ANNPredictor2(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, p=0.5, use_last_year_smb=False, **args):
        super(ANNPredictor2, self).__init__()
        self.args = args
        self.last_year = use_last_year_smb
        if use_last_year_smb:
            input_dim = input_dim + 1
        self.predictor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(in_features=hidden_dim // 4, out_features=hidden_dim // 8),
            nn.Dropout(p=p),
            nn.Linear(in_features=hidden_dim // 8, out_features=1)
        )

    def forward(self, x, last_dmdt=None):
        if self.last_year:
            assert last_dmdt is not None
            assert last_dmdt.shape == (x.shape[0], 1)
            x = torch.cat([x, last_dmdt], dim=1)
        return self.predictor(x)
