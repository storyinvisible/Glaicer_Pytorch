import os.path
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset
from models import GlacierModel, LSTMPredictor, VCNN, SeparateFeatureExtractor
from utils import plot_loss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import runner


def train(name, train_loader, test_dataset, test_smb, split_at):
    # construct the model
    # vcnn_model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
    lstm_model = LSTMPredictor(layers=None, input_dim=256, hidden_dim=[256, 128, 64, 32], n_layers=1, bidirection=False,
                               p=0.5)
    extractor = SeparateFeatureExtractor(output_dim=256, layers=[
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
    ])

    glacier_model = GlacierModel(extractor, lstm_model, name="sepVCNNLSTM{:.4}".format(name))

    # train model
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss()
    runner.trainer(glacier_model, train_loader=train_loader, testdataset=test_dataset, testsmb=test_smb,
                   show=False,
                   device=cuda, epochs=150, lr=0.002, reg=0.001, save_every=20, eval_every=1, test_split_at=split_at,
                   critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")


glaicer_selected = pd.read_csv("Glacier_select.csv")

glaciers = list(glaicer_selected["NAME"])
end_year = 2018
for name in glaciers:
    cond_1 = glaicer_selected["NAME"] == name
    years = int(list(glaicer_selected[cond_1]["Years"])[0])
    cal_start_year = lambda years: (2018 - years) if ((2018 - years) > 1979) else 1979
    start_year = cal_start_year(years)
    test_years = int(years * 0.2)
    train_smb = Glacier_dmdt(name, start_year, end_year - test_years, path="glacier_dmdt.csv")
    train_data = ERA5Datasets(name, start_year, end_year - test_years, path="ECMWF_reanalysis_data")
    train_dataset = GlacierDataset([train_data], [train_smb])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    second = train_data[1].shape[1]

    if second <48:
        continue
    test_smb = Glacier_dmdt(name, end_year - test_years, end_year, path="glacier_dmdt.csv")
    test_data = ERA5Datasets(name, end_year - test_years, end_year, path="ECMWF_reanalysis_data")
    test_dataset = GlacierDataset([test_data], [test_smb])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train(name, train_loader, test_data, test_smb,int(years * 0.8))

# print(glaciers)
# JAKOBSHAVN_smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="glacier_dmdt.csv")
# JAKOBSHAVN_data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="ECMWF_reanalysis_data")
# glacier_dataset = GlacierDataset([JAKOBSHAVN_data], [JAKOBSHAVN_smb])
# loader = DataLoader(glacier_dataset, batch_size=16, shuffle=True)
