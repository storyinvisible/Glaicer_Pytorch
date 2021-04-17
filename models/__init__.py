import torch.nn as nn
from models.predictors.ANNPredictor import ANNPredictor
from models.predictors.LSTMPredictor import LSTMPredictor
from models.extractor.hcnn import HCNN
from models.extractor.vcnn import VCNN


class GlacierModel(nn.Module):
    def __init__(self, extra, pred):
        super(GlacierModel, self).__init__()
        self.extra = extra
        self.pred = pred

    def forward(self, x):
        out = self.extra(x)
        out = self.pred(out)
        return out


class Predictor(nn.Module):
    def __init__(self, layers=None, input_dim=256, hidden_dim=256, n_layers=1, bi_direction=False, p=0.5):
        super(Predictor, self).__init__()
        if layers is None:
            raise ValueError("[ERROR] The layers cannot be empty")
        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)
