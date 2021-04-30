import torch.nn as nn
from models.predictors.ANNPredictor import ANNPredictor
from models.predictors.LSTMPredictor import LSTMPredictor
from models.extractor.hcnn import HCNN
from models.extractor.vcnn import VCNN
from copy import deepcopy
import torch
class GlacierModel(nn.Module):
    def __init__(self, extra, pred):
        super(GlacierModel, self).__init__()
        self.extra = extra
        self.pred = pred

    def forward(self, x):
        out = self.extra(x)
        out = self.pred(out)
        return out

class GlacierModel2(nn.Module):
    def __init__(self, extra, pred):
        super(GlacierModel2, self).__init__()
        self.extra1 = extra
        self.extra2 = deepcopy(extra)
        self.extra3 = deepcopy(extra)
        self.extra4 = deepcopy(extra)
        self.extra5 = deepcopy(extra) # the five features needed
        self.pred = pred

    def forward(self, x):
        x1 = torch.unsqueeze(x[:,0,:,:],0)
        x2 = torch.unsqueeze(x[:, 1, :, :], 0)
        x3 = torch.unsqueeze(x[:, 2, :, :], 0)
        x4 = torch.unsqueeze(x[:, 3, :, :], 0)
        x5 = torch.unsqueeze(x[:, 4, :, :], 0)
        out1=self.extra1(x1)
        out2 = self.extra2(x2)
        out3 = self.extra3(x3)
        out4 = self.extra4(x4)
        out5 = self.extra5(x5)

        out = torch.cat((out1,out2,out3,out4,out5), dim=1)
        print(out.shape)
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
