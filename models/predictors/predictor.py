import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, layers=None, input_dim=256, hidden_dim=256, n_layers=1, bi_direction=False, p=0.5):
        super(Predictor, self).__init__()
        if layers is None:
            raise ValueError("[ERROR] The layers cannot be empty")
        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)
