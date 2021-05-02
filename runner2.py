import torch
import os.path
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset, GlacierDataset3D
from models import GlacierModel, ANNPredictor, ANNPredictor2, LSTMPredictor, Predictor, HCNN, VCNN, TCNN, TWCNN
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
    train_pred, train_actual = [], []
    for epoch in range(1, epochs + 1):
        total_train_loss = 0
        count = 0
        for feature, target in train_loader:
            feature, target = Variable(feature.type(torch.FloatTensor)).to(device), Variable(target).to(device)
            pred = model(feature)
            train_pred.append(pred.item())
            train_actual.append(target.float())
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
                prediction_plot, predicted, actual = evaluate(model, train_pred, train_actual, testdataset, device=device)
                prediction_plot.savefig("plots/{}_{}_pred_and_actual.png".format(model.name, testdataset.glacier_name))
                prediction_plot.close()
                test_loss = critic(torch.tensor([predicted]), torch.tensor([actual])).item()

                test_losses.append(test_loss)
                mean_loss = total_train_loss / eval_every
                train_losses.append(mean_loss)
                print(
                    "[INFO] Epoch {}|{} {} Loss: {:.4f} Eval: {:.4f}".format(epoch, epochs, step, mean_loss, test_loss))
                loss_plot = plot_loss(train_losses, test_losses, show=show)
                loss_plot.savefig("plots/{}_{}_loss_plot.png".format(model.name, testdataset.glacier_name))
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


def evaluate(model, train_pred, train_actual, dataset, device=None):
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
    pred = train_pred.extend(result)
    actual = train_actual.extend(real)
    print(pred, actual)
    plt.figure()
    year_range = np.arange(dataset.start_year, dataset.end_year)
    predicted, = plt.plot(year_range, pred, color="blue", linestyle='--')
    actual, = plt.plot(year_range, actual, color="red", linewidth=2)
    split_at = dataset.start_year + len(train_pred)
    min_value, max_value = min(min(pred), min(actual)), max(max(pred), max(actual))
    plt.vlines(split_at, ymin=min_value, ymax=max_value, color="green", linestyle='--')
    plt.ylabel("dm/dt")
    plt.xlabel("year")
    plt.legend([actual, predicted], ["Actual", "Predict"], loc="upper left")
    return plt, result, real

def train_val_dataset(dataset, val_split=0.2):
    split_idx = int(len(dataset)*val_split)
    train_idx, val_idx = list(range(len(dataset)-split_idx)), list(range(len(dataset)-split_idx, len(dataset)))
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['val'].end_year = dataset.end_year
    datasets['val'].start_year = dataset.start_year
    datasets['val'].glacier_name = dataset.glacier_name
    return datasets

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glacier_info = pd.read_csv("Glacier_select.csv")
    glaciers =list(glacier_info["NAME"])
    for name in glaciers:
        new_df = glacier_info[glacier_info["NAME"]==name]
        start_year = 2018 - int(new_df["Years"])
        if start_year < 1979:
            start_year = 1979
        dataset = GlacierDataset3D(name, start_year, 2018, path="glacier_dmdt.csv")
        datasets = train_val_dataset(dataset)
        train_loader = DataLoader(datasets['train'], batch_size=1)
        test_dataset = datasets['val']
        # tcnn_model = TCNN()
        twcnn_model = TWCNN()
        lstm_model = LSTMPredictor(layers=None, input_dim=224, hidden_dim=224, n_layers=1, bidirection=False, p=0.5)
        ann_model = ANNPredictor2(layers=None, input_dim=224, hidden_dim=224, n_layers=1, bidirection=False, p=0.5)
        glacier_model = GlacierModel(twcnn_model, lstm_model, name="TWCNNLSTM3D")
        cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_function = torch.nn.MSELoss()
        trainer(glacier_model, train_loader=train_loader, testdataset=test_dataset, show=False,
                device=cuda, epochs=20, lr=0.002, reg=0.001, save_every=10, eval_every=1, test_split_at=15,
                critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")
