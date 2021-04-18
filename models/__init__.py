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
    def __init__(self, layers=None, **args):
        super(Predictor, self).__init__()
        self.args = args
        if layers is None:
            raise ValueError("[ERROR] The layers cannot be empty")
        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)
