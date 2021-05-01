import torch
import os.path
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset, GlacierDataset3D
from models import GlacierModel, ANNPredictor, LSTMPredictor, Predictor, HCNN, VCNN, TCNN, TWCNN
from utils import plot_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trainer(model, train_loader, testdataset, critic, optimizer, epochs=500, lr=0.002,
            reg=0.001, save_every=10, eval_every=10, save_path=None, show=False, device=None, test_split_at=None):
    device = torch.device("cpu") if device is None else device
    optim = optimizer(model.parameters(), lr=lr, weight_decay=reg)
    model = model.to(device)
    step = 0
    train_losses, test_losses = [], []
    for epoch in range(1, epochs + 1):
        total_train_loss = 0
        count = 0
        for feature, target in train_loader:
            feature, target = Variable(feature.type(torch.FloatTensor)).to(device), Variable(target).to(device)
            pred = model(feature)
            loss = critic(pred.squeeze(1), target.float())
            optim.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optim.step()
            total_train_loss += loss.item()
            step += 1
            count += 1
            if step % save_every == 0:
                torch.save(model, os.path.join(save_path, "{}_model.h5".format(model.name)))
            if step % eval_every == 0:
                prediction_plot, predicted, actual = evaluate(model, testdataset,
                                                              split_at=test_split_at, device=device)
                prediction_plot.savefig("plots/{}_{}_pred_and_actual.png".format(model.name, testdataset.glacier_name))
                prediction_plot.close()
                test_loss = critic(torch.tensor([predicted]), torch.tensor([actual])).item()

                test_losses.append(test_loss)
                mean_loss = total_train_loss / eval_every
                train_losses.append(mean_loss)
                print(
                    "[INFO] Epoch {}|{} {} Loss: {:.4f} Eval: {:.4f}".format(epoch, epochs, step, mean_loss, test_loss))
                loss_plot = plot_loss(train_losses, test_losses, show=show)
                loss_plot.savefig("plots/{}_loss_plot.png".format(model.name))
                loss_plot.close()


def predict(model, dataset, device=None):
    with torch.no_grad():
        device = torch.device("cpu") if device is None else device
        model = model.to(device)
        result = []
        try:
            for data in dataset:
                pred = model(data.to(device))
                result.append(pred.item())
        except KeyError:
            pass
    return result


def evaluate(model, dataset, split_at=None, device=None):
    with torch.no_grad():
        device = torch.device("cpu") if device is None else device
        result, real = [], []
        try:
            for data, t in dataset:
                pred = model(torch.from_numpy(data).unsqueeze(0).float().to(device))
                result.append(pred.item())
                real.append(t)
        except Exception as e:
            print(e)
    plt.figure()
    year_range = np.arange(dataset.start_year, dataset.end_year)
    predicted, = plt.plot(year_range, result, color="blue", linestyle='--')
    actual, = plt.plot(year_range, real, color="red", linewidth=2)
    if split_at:
        if split_at < 1800:
            split_at = split_at + dataset.start_year
        min_value, max_value = min(min(result), min(real)), max(max(result), max(real))
        plt.vlines(split_at, ymin=min_value, ymax=max_value, color="green", linestyle='--')
    plt.ylabel("dm/dt")
    plt.xlabel("year")
    plt.legend([actual, predicted], ["Actual", "Predict"], loc="upper left")
    return plt, result, real


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glacier_info = pd.read_csv("Glacier_select.csv")
    new_df = glacier_info[glacier_info["NAME"]=="JAKOBSHAVN_ISBRAE"]
    start_year = 2018 - int(new_df["Years"])
    if start_year < 1979:
        start_year = 1979
    # smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", start_year, 2018, path="glacier_dmdt.csv")
    # data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="ECMWF_reanalysis_data")
    # dataset = GlacierDataset([data], [smb])
    dataset = GlacierDataset3D("JAKOBSHAVN_ISBRAE", start_year, 2018, path="glacier_dmdt.csv")
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    tcnn_model = TCNN()
    twcnn_model = TWCNN()
    lstm_model = LSTMPredictor(layers=None, input_dim=224, hidden_dim=224, n_layers=1, bidirection=False, p=0.5)
    ann_model = ANNPredictor(layers=None, input_dim=224, hidden_dim=224, n_layers=1, bidirection=False, p=0.5)
    predictor_model = Predictor(input_dim=224, hidden_dim=224, n_layers=2, bidirection=True, p=0.5, layers=[
        torch.nn.Linear(224, 224),
        torch.nn.ReLU(),
        torch.nn.Linear(224, 1)
    ])
    glacier_model = GlacierModel(twcnn_model, lstm_model, name="TWCNNLSTM3D")
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss()
    trainer(glacier_model, train_loader=train_loader, testdataset=dataset, show=False,
            device=cuda, epochs=100, lr=0.002, reg=0.001, save_every=10, eval_every=1, test_split_at=15,
            critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")
