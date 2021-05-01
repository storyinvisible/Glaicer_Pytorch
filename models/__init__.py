import torch
import torch.nn as nn
from models.predictors.ANNPredictor import ANNPredictor
from models.predictors.LSTMPredictor import LSTMPredictor
from models.extractor.hcnn import HCNN
from models.extractor.vcnn import VCNN
from models.extractor.tcnn import TCNN
from models.extractor.twcnn import TWCNN
from models.extractor.hinverted import HInvertedBlock, HInvertedResidual
from models.extractor.vinverted import VInvertedBlock, VInvertedResidual


class GlacierModel(nn.Module):
    def __init__(self, extra, pred, name):
        super(GlacierModel, self).__init__()
        self.extra = extra
        self.pred = pred
        self.name = name

    def forward(self, x):
        out = self.extra(x)
        out = self.pred(out)
        return out


class FlattenInputForSeparateModel(nn.Module):
    def __init__(self, concate=False):
        super(FlattenInputForSeparateModel, self).__init__()
        self.concat = concate

    def forward(self, x):
        out = [torch.unsqueeze(x[:, i, :, :], 1) for i in range(x.shape[1])]
        if self.concat:
            return torch.cat(out, dim=3)
        return out


class SeparateFeatureExtractor(nn.Module):
    def __init__(self, in_channel=None, output_dim=256, **args):
        super(SeparateFeatureExtractor, self).__init__()
        self.layers = args["layers"]
        self.output_dim = output_dim
        self.flattener = FlattenInputForSeparateModel()
        hidden = sum([l.output_dim for l in self.layers])
        self.output = nn.Linear(hidden, output_dim)

    def _apply(self, fn):
        for layer in self.layers:
            layer._apply(fn)
        self.output._apply(fn)

    def forward(self, x):
        features = self.flattener(x)
        result = []
        for model, data in zip(self.layers, features):
            result.append(model(data))
        out = torch.cat(result, dim=-1)
        return self.output(out)


class Predictor(nn.Module):
    def __init__(self, layers=None, **args):
        super(Predictor, self).__init__()
        self.args = args
        if layers is None:
            raise ValueError("[ERROR] The layers cannot be empty")
        self.predictor = nn.Sequential(*layers)

    def forward(self, x):
        return self.predictor(x)
