import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset
from models import GlacierModel, ANNPredictor, LSTMPredictor, Predictor, HCNN, VCNN
from utils import plot_loss, plot_actual
import numpy as np


def trainer(extractor, predictor, train_loader, test_loader, dataset, loss_func, optimizer, epochs=500, lr=0.002,
            reg=0.001,
            save_every=10, print_every=10, save_path=None, show=False, device=None):
    model = GlacierModel(extractor, predictor).to(device)
    critic = loss_func()
    optim = optimizer(model.parameters(), lr=lr, weight_decay=reg)
    step = 0
    train_loss, test_loss = [], []
    pred_train = None
    pred_test = None
    for epoch in range(epochs):
        pred_train = None
        pred_test = None
        total_train_loss = 0
        for feature, target in train_loader:
            feature, target = Variable(feature).to(device), Variable(target).to(device)
            step += 1
            pred = model(feature)
            if pred_train is None:
                pred_train = pred.cpu().detach().numpy()[0]
            else:
                pred_train = np.append(pred_train, pred.cpu().detach().numpy()[0])
            loss = critic(pred.squeeze(1), target.float())
            optim.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optim.step()
            total_train_loss += loss.item()
            if step % print_every == 0:
                print("[INFO] Epoch {}|{}|{} Loss :{:.4f}".format(step, epoch, epochs, total_train_loss / print_every))
            if step % save_every == 0:
                torch.save(model, save_path)
        train_loss.append(total_train_loss / print_every)
        total_test_loss = 0
        step = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature, target = feature.to(device), target.to(device)
                pred = model(feature)
                loss = critic(pred.squeeze(1), target.float().to(device))
                if pred_test is None:
                    pred_test = pred.cpu().detach().numpy()[0]
                else:
                    pred_test = np.append(pred_test, pred.cpu().detach().numpy()[0])
                total_test_loss += loss.item()
                step += 1
        print("[INFO] Epoch {}|{} Loss :{:.4f}".format(epoch, epochs, total_test_loss / step))
    loss_plot = plot_loss(train_loss, test_loss, show=show)
    loss_plot.savefig("Plot/loss_plot.png")
    loss_plot.close()
    act_plt = plot_actual(smb, pred_train, pred_test)
    act_plt.savefig("Plot/actual_plot.png")
    act_plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="glaicer_dmdt.csv")
    data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="ECMWF_reanalysis_data")
    dataset = GlacierDataset([data], [smb])
    train_loader = DataLoader(dataset, batch_size=1)
    test_loader = DataLoader(dataset, batch_size=1)
    vcnn_model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
    hcnn_model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
    lstm_model = LSTMPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
    ann_model = ANNPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
    predictor_model = Predictor(input_dim=256, hidden_dim=256, n_layers=2, bidirection=True, p=0.5, layers=[
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1)
    ])
    trainer(vcnn_model, lstm_model, train_loader=train_loader, test_loader=test_loader, dataset=smb, show=False,
            device=device, epochs=3, lr=0.002, reg=0.001, save_every=10, print_every=10,
            loss_func=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            save_path="saved_models/HCNN_LSTM_model.h5")
