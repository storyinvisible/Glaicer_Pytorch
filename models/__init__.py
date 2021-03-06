import torch
import torch.nn as nn

from models.extractor.hcnn import HCNN
from models.extractor.hinverted import HInvertedBlock, HInvertedResidual
from models.extractor.tcnn import TCNN
from models.extractor.twcnn import TWCNN, TWCNN2D
from models.extractor.vcnn import VCNN
from models.extractor.vinverted import VInvertedBlock, VInvertedResidual
from models.predictors.ANNPredictor import ANNPredictor, ANNPredictor2
from models.predictors.LSTMPredictor import LSTMPredictor, LSTMPredictor3D


class GlacierModel(nn.Module):
    def __init__(self, extra, pred, name, use_last_year_smb=False):
        super(GlacierModel, self).__init__()
        self.extra = extra
        self.pred = pred
        self.name = name
        self.last_year = use_last_year_smb

    def forward(self, x, last_dmdt=None):
        if self.last_year:
            out = self.extra(x)
            return self.pred(out, last_dmdt)
        out = self.extra(x)
        return self.pred(out)


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
        out = self.output(out)
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


class FlattenInputForSeparateModel3D(nn.Module):
    def __init__(self, concate=False):
        super(FlattenInputForSeparateModel3D, self).__init__()
        self.concat = concate

    def forward(self, x):
        out = [x[:, i, :, :] for i in range(x.shape[1])]
        if self.concat:
            return torch.cat(out, dim=3)
        return out


class SeparateFeatureExtractor3D(nn.Module):
    def __init__(self, in_channel=None, output_dim=256, **args):
        super(SeparateFeatureExtractor3D, self).__init__()
        self.layers = args["layers"]
        self.output_dim = output_dim
        self.flattener = FlattenInputForSeparateModel3D()
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
        out = self.output(out)
        return out
