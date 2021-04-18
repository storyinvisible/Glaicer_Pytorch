import torch.nn as nn


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
