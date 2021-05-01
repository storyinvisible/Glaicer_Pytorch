import os.path

import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset
from models import GlacierModel, LSTMPredictor, VCNN, SeparateFeatureExtractor
from utils import plot_loss
import numpy as np
import matplotlib.pyplot as plt


def trainer(model,name, train_loader, testdataset, testsmb, critic, optimizer, epochs=500, lr=0.002,
            reg=0.001, save_every=10, eval_every=10, save_path=None, show=False, device=None, test_split_at=None):
    device = torch.device("cpu") if device is None else device
    optim = optimizer(model.parameters(), lr=lr, weight_decay=reg)
    model = model.to(device)
    step = 0
    train_losses, test_losses = [], []
    base_path = os.path.join(save_path, model.name)
    best_train_loss= 1000000
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
                if step % save_every == 0:
                    torch.save(model, os.path.join(base_path, "{}_model-{}.h5".format(model.name, epoch)))
                if step % eval_every == 0:
                    prediction_plot, predicted, actual = evaluate(model, testdataset, testsmb,
                                                                  split_at=test_split_at, device=device)
                    test_loss = critic(torch.tensor([predicted]), torch.tensor([actual])).item() / min(len(testdataset),
                                                                                                         len(testsmb))
                    if test_loss <best_train_loss:
                        best_train_loss=test_loss
                    test_losses.append(test_loss)
                    mean_loss = total_train_loss / eval_every / train_loader.batch_size
                    train_losses.append(mean_loss)
                    prediction_plot.savefig(
                        "{}/comp-{}_{:.4f}_{:.4f}.png".format(os.path.join(base_path, "plots"), epoch,
                                                              loss.item() / train_loader.batch_size,
                                                              test_loss))
                    prediction_plot.close()
                    print("[INFO] Epoch {}|{} {} Loss: {:.4f} Eval: {:.4f}".format(epoch, epochs, step, mean_loss,
                                                                                   test_loss))
                    loss_plot = plot_loss(train_losses, test_losses, show=show)
                    loss_plot.savefig("{}/{}_loss.png".format(os.path.join(base_path, "plots"), model.name))
                    loss_plot.close()
    except KeyboardInterrupt:
        print("[INFO] Starting to exit!")
    finally:
        # save model
        torch.save(model, os.path.join(base_path, "{}_model.h5".format(model.name)))
        # loss plot
        loss_plot = plot_loss(train_losses, test_losses, show=show)
        loss_plot.savefig("{}/{}_loss.png".format(os.path.join(base_path, "plots"), model.name))
        loss_plot.close()
        # loss record
        pd.DataFrame({"train_loss": train_losses, "eval_loss": test_losses}).to_csv(os.path.join(base_path, "loss.csv"))
        # Final evaluation
        prediction_plot, predicted, actual = evaluate(model, testdataset, testsmb, split_at=test_split_at,
                                                      device=device)
        prediction_plot.savefig("{}/{}_{}_comp.png".format(os.path.join(base_path, "plots"), model.name,
                                                           testdataset.glacier))
        prediction_plot.close()
    filename= "Loss/loss_summary"+model.name+".csv"
    if os.path.exists(filename):
        loss_df=pd.read_csv(filename)
        loss_df[name]=[best_train_loss]
        loss_df.to_csv(filename)
    else:
        if os.path.exists("Loss"):
            os.mkdir("Loss")
        loss_df=pd.DataFrame({name:[best_train_loss]})
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
    JAKOBSHAVN_smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2010, path="glacier_dmdt.csv")
    JAKOBSHAVN_data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2010, path="ECMWF_reanalysis_data")
    JAKOBSHAVN_smb_test = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2018, path="glacier_dmdt.csv")
    JAKOBSHAVN_data_test = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2018, path="ECMWF_reanalysis_data")
    # QAJUUTTAP_SERMIA_smb = Glacier_dmdt("QAJUUTTAP_SERMIA", 1980, 2002, path="glaicer_dmdt.csv")
    QAJUUTTAP_SERMIA_data = ERA5Datasets("QAJUUTTAP_SERMIA", 1980, 2002, path="ECMWF_reanalysis_data")
    # STORSTROMMEN_smb = Glacier_dmdt("STORSTROMMEN", 1980, 2002, path="glaicer_dmdt.csv")
    STORSTROMMEN_data = ERA5Datasets("STORSTROMMEN", 1980, 2002, path="ECMWF_reanalysis_data")
    # HELHEIMGLETSCHER_smb = Glacier_dmdt("HELHEIMGLETSCHER", 1980, 2002, path="glaicer_dmdt.csv")
    HELHEIMGLETSCHER_data = ERA5Datasets("HELHEIMGLETSCHER", 1980, 2002, path="ECMWF_reanalysis_data")
    # DAUGAARD_smb = Glacier_dmdt("DAUGAARD-JENSEN", 1980, 2002, path="glaicer_dmdt.csv")
    DAUGAARD_data = ERA5Datasets("DAUGAARD-JENSEN", 1980, 2002, path="ECMWF_reanalysis_data")
    glacier_dataset = GlacierDataset([JAKOBSHAVN_data], [JAKOBSHAVN_smb])
    loader = DataLoader(glacier_dataset, batch_size=16, shuffle=False)
    """
    QAJ torch.Size([5, 14, 12])
    STO torch.Size([5, 326, 12])
    HEL torch.Size([5, 167, 12])
    DAU torch.Size([5, 210, 12])
    """
    # construct the model
    # vcnn_model = VCNN(in_channel=5, output_dim=256, vertical_dim=289)
    lstm_model = LSTMPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
    extractor = SeparateFeatureExtractor(output_dim=256, layers=[
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
        VCNN(in_channel=1, output_dim=256, vertical_dim=289),
    ])

    glacier_model = GlacierModel(extractor, lstm_model, name="separateVCNNLSTM")

    # train model
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.MSELoss()
    trainer(glacier_model, train_loader=loader, testdataset=JAKOBSHAVN_data_test, testsmb=JAKOBSHAVN_smb_test,
            show=False,
            device=cuda, epochs=300, lr=0.002, reg=0.001, save_every=1, eval_every=1, test_split_at=2010,
            critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")
