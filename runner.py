import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset
from models import GlacierModel, LSTMPredictor, VCNN, HCNN
from utils import plot_loss


def trainer(model, train_loader, testdataset, testsmb, critic, optimizer, epochs=500, lr=0.002,
            reg=0, save_every=10, eval_every=10, save_path=None, show=False, device=None, test_split_at=None,
            best_only=True):
    device = torch.device("cpu") if device is None else device
    optim = optimizer(model.parameters(), lr=lr, weight_decay=reg)
    model = model.to(device)
    step = 0
    train_losses, test_losses = [], []
    base_path = os.path.join(save_path, model.name)
    best_train_loss = 1E5
    test_loss = 1E5
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(os.path.join(base_path, "plots")):
        os.makedirs(os.path.join(base_path, "plots"))
    try:
        for epoch in range(1, epochs + 1):
            total_train_loss = 0
            count = 0
            for feature, target in train_loader:
                feature, target = Variable(feature.type(torch.FloatTensor)).to(device), Variable(
                    target.type(torch.FloatTensor)).to(device)
                pred = model(feature)
                loss = critic(pred.squeeze(1), target.float())
                optim.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optim.step()
                total_train_loss += loss.item()
                step += 1
                count += 1
                if step % eval_every == 0:
                    prediction_plot, predicted, actual = evaluate(model, testdataset, testsmb,
                                                                  split_at=test_split_at, device=device)
                    test_loss = critic(torch.tensor([predicted]), torch.tensor([actual])).item() / min(len(testdataset),
                                                                                                       len(testsmb))
                    test_losses.append(test_loss)
                    mean_loss = total_train_loss / eval_every / train_loader.batch_size
                    train_losses.append(mean_loss)
                    calc_loss = mean_loss * (test_loss ** 2)
                    if best_only:
                        if calc_loss <= best_train_loss:
                            prediction_plot.savefig(
                                "{}/pred_and_act_{}.png".format(os.path.join(base_path, "plots"), testdataset.glacier))
                    prediction_plot.close()
                    if calc_loss < best_train_loss:
                        best_train_loss = calc_loss
                    print("[INFO] Epoch {}|{} {} Loss: {:.4f} Eval: {:.4f}".format(epoch, epochs, step, mean_loss,
                                                                                   test_loss))
                    loss_plot = plot_loss(train_losses, test_losses, show=show)
                    loss_plot.savefig("{}/{}_loss.png".format(os.path.join(base_path, "plots"), model.name))
                    loss_plot.close()
                if step % save_every == 0:
                    if best_only:
                        mean_loss = total_train_loss / eval_every / train_loader.batch_size
                        if mean_loss * (test_loss ** 2) <= best_train_loss:
                            torch.save(model, os.path.join(base_path, "{}_model.h5".format(model.name)))
                    else:
                        torch.save(model, os.path.join(base_path, "{}_model-{}.h5".format(model.name, epoch)))
    except KeyboardInterrupt:
        print("[INFO] Starting to exit!")
    finally:
        # save model
        torch.save(model, os.path.join(base_path, "{}_model_final.h5".format(model.name)))
        # loss plot
        loss_plot = plot_loss(train_losses, test_losses, show=show)
        loss_plot.savefig("{}/{}_final_loss.png".format(os.path.join(base_path, "plots"), model.name))
        loss_plot.close()
        # loss record
        pd.DataFrame({"train_loss": train_losses, "eval_loss": test_losses}).to_csv(os.path.join(base_path, "loss.csv"))
        # Final evaluation
        prediction_plot, predicted, actual = evaluate(model, testdataset, testsmb, split_at=test_split_at,
                                                      device=device)
        prediction_plot.savefig("{}/{}_{}_final_comp.png".format(os.path.join(base_path, "plots"), model.name,
                                                                 testdataset.glacier))
        prediction_plot.close()
        if not os.path.exists(os.path.join(save_path, "Loss")):
            os.makedirs(os.path.join(save_path, "Loss"))
        filename = os.path.join(save_path, "Loss", "loss.csv")
        if os.path.exists(filename):
            loss_df = pd.read_csv(filename)
            loss_df[model.name] = [best_train_loss]
            loss_df.to_csv(filename)
        else:

            loss_df = pd.DataFrame({model.name: [best_train_loss]})
            loss_df.to_csv(filename)


def predict(model, dataset, device=None):
    with torch.no_grad():
        device = torch.device("cpu") if device is None else device
        model = model.to(device)
        result = []
        try:
            for data in dataset:
                pred = model(data.unsqueeze(0).to(device))
                result.append(pred.item())
        except KeyError:
            pass
    return result


def evaluate(model, dataset, target, split_at=None, device=None):
    with torch.no_grad():
        device = torch.device("cpu") if device is None else device
        result, real = [], []
        try:
            for data, t in zip(dataset, target):
                pred = model(data.unsqueeze(0).to(device))
                result.append(pred.item())
                real.append(t)
        except KeyError:
            pass
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
    # set dataset
    glacier_name = "STORSTROMMEN"
    start, mid, end = 1979, 2008, 2018
    train_smb = Glacier_dmdt(glacier_name, start, mid, path="glacier_dmdt.csv")
    train_data = ERA5Datasets(glacier_name, start, mid, path="ECMWF_reanalysis_data")
    test_smb = Glacier_dmdt(glacier_name, start, end, path="glacier_dmdt.csv")
    test_data = ERA5Datasets(glacier_name, start, end, path="ECMWF_reanalysis_data")
    glacier_dataset = GlacierDataset([train_data], [train_smb])
    loader = DataLoader(glacier_dataset, batch_size=16, shuffle=False)
    # QAJ torch.Size([5, 14, 12])
    # STO torch.Size([5, 326, 12])
    # HEL torch.Size([5, 167, 12])
    # DAU torch.Size([5, 210, 12])
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss()

    lstm_model = LSTMPredictor(layers=None, input_dim=[256, 128, 64, 32], hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
    extractor = HCNN(in_channel=5, output_dim=256, vertical_dim=326)
    glacier_model = GlacierModel(extractor, lstm_model, name="HCNNLSTM-STOR")
    trainer(glacier_model, train_loader=loader, testdataset=test_data, testsmb=test_smb,
            show=False, device=cuda, epochs=250, lr=0.002, reg=0, save_every=1, eval_every=1,
            test_split_at=mid, best_only=True,
            critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")

    lstm_model = LSTMPredictor(layers=None, input_dim=[256, 128, 64, 32], hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
    extractor = VCNN(in_channel=5, output_dim=256, vertical_dim=326)
    glacier_model = GlacierModel(extractor, lstm_model, name="VCNNLSTM-STOR")
    trainer(glacier_model, train_loader=loader, testdataset=test_data, testsmb=test_smb,
            show=False, device=cuda, epochs=250, lr=0.002, reg=0, save_every=1, eval_every=1,
            test_split_at=mid, best_only=True,
            critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")
