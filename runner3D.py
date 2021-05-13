import torch
import os.path
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset, GlacierDataset3D, GlacierDatasetNoPadding3D
from models import GlacierModel, ANNPredictor, ANNPredictor2, LSTMPredictor, LSTMPredictor3D, Predictor, TCNN, TWCNN, TWCNN2D, SeparateFeatureExtractor3D
from utils import plot_loss, plot_smb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def trainer(model, train_loader, testdataset, critic, optimizer, epochs=500, lr=0.002,
            reg=0.001, save_every=10, eval_every=10, save_path=None, show=False, test_last_year_dmdt=None, use_last_year=False, device=None, test_split_at=None, 
            best_only=True):
    device = torch.device("cpu") if device is None else device
    optim = optimizer(model.parameters(), lr=lr, weight_decay=reg)
    model = model.to(device)
    base_path = os.path.join(save_path, model.name)
    best_train_loss = 1E7
    test_loss = 1E7
    step = 0
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    if not os.path.exists(os.path.join(base_path, "plots")):
        os.makedirs(os.path.join(base_path, "plots"))
    train_losses, test_losses = [], []
    try:
        for epoch in range(1, epochs + 1):
            total_train_loss = 0
            count = 0
            train_pred, train_actual = [], []
            for feature, target in train_loader:
                if use_last_year:
                    feature, last_year_dmdt = feature
                    feature = Variable(feature.type(torch.FloatTensor)).to(device)
                    last_year_dmdt = Variable(last_year_dmdt.type(torch.FloatTensor)).to(device).unsqueeze(1)
                    target = Variable(target.type(torch.FloatTensor)).to(device)
                    pred = model(feature, last_year_dmdt)
                else:
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
                if step % eval_every == 0:
                    predicted, actual = evaluate(model, testdataset, last_year_dmdt=test_last_year_dmdt, 
                                                use_last_year=use_last_year, device=device)
                    test_loss = critic(torch.tensor([predicted]), torch.tensor([actual])).item() / len(testdataset)
                    test_losses.append(test_loss)
                    mean_loss = total_train_loss / count / train_loader.batch_size
                    train_losses.append(mean_loss)
                    cal_loss = mean_loss*(test_loss**2)
                    if cal_loss < best_train_loss:
                        best_train_loss = cal_loss
                    print("[INFO] Epoch {}|{} {} Loss: {:.4f} Eval: {:.4f}".format(epoch, epochs, step, mean_loss,
                                                                                   test_loss))
                    loss_plot = plot_loss(train_losses, test_losses, show=show)
                    loss_plot.savefig("{}/{}_{}_loss.png".format(os.path.join(base_path, "plots"), testdataset.glacier, model.name))
                    loss_plot.close()
    except KeyboardInterrupt:
        print("[INFO] Starting to exit!")
    except Exception as e:
        print(e)
    finally:
        # save model
        torch.save(model, os.path.join(base_path, "{}_model.h5".format(model.name)))
        # loss plot
        loss_plot = plot_loss(train_losses, test_losses, show=show)
        loss_plot.savefig("{}/{}_loss.png".format(os.path.join(base_path, "plots"), model.name))
        loss_plot.close()
        # loss record
        pd.DataFrame({"train_loss": train_losses, "eval_loss": test_losses}).to_csv(os.path.join(base_path, "loss_{}.csv".format(testdataset.glacier)))
        # Final evaluation
        predicted, actual = evaluate(model, testdataset, last_year_dmdt=test_last_year_dmdt, 
                                                use_last_year=use_last_year, device=device)
        prediction_plot = plot_smb(train_actual, actual, train_pred, predicted, testdataset.start_year, testdataset.start_year+len(train_loader)-1)
        prediction_plot.savefig("{}/{}_{}_comp.png".format(os.path.join(base_path, "plots"), model.name,
                                                           testdataset.glacier))
        prediction_plot.close()
        if not os.path.exists(os.path.join(save_path, "Loss")):
            os.makedirs(os.path.join(save_path, "Loss"))
        filename = os.path.join(save_path, "Loss", "loss_summary_{}.csv".format(model.name))
        if os.path.exists(filename):
            loss_df = pd.read_csv(filename)
            loss_df[test_dataset.glacier] = [best_train_loss]
            loss_df.to_csv(filename)
        else:
            if os.path.exists("Loss"):
                os.mkdir("Loss")
            loss_df = pd.DataFrame({test_dataset.glacier: [best_train_loss]})
            loss_df.to_csv(filename)


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


def evaluate(model, dataset, last_year_dmdt=None, use_last_year=False, device=None):
    with torch.no_grad():
        device = torch.device("cpu") if device is None else device
        result, real = [], []
        if use_last_year:
            for d, last_dmdt in zip(DataLoader(dataset, batch_size=1), DataLoader(last_year_dmdt, batch_size=1)):
                d, t = d
                data = torch.tensor(d[0]).float().to(device)
                last_dmdt = Variable(d[1]).unsqueeze(1).float().to(device)
                pred = model(data, last_dmdt)
                result.append(pred.item())
                real.append(t)
        else:
            for data, t in dataset:
                data = torch.from_numpy(data).unsqueeze(0).float().to(device)
                print(data.shape)
                pred = model(data)
                result.append(pred.item())
                real.append(t)
    return result, real

def train_val_dataset(dataset, val_split=0.3, shuffle=False):
    split_idx = len(dataset) - int(len(dataset)*val_split)
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    train_idx, val_idx = indices[:split_idx], indices[split_idx-1:]
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    datasets['val'].end_year = dataset.end_year
    datasets['val'].start_year = dataset.start_year
    datasets['val'].glacier = dataset.glacier
    return datasets

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    glacier_info = pd.read_csv("Glacier_select.csv")
    glaciers =list(glacier_info["NAME"])
    for name in glaciers:
        try:
            new_df = glacier_info[glacier_info["NAME"]==name]
            start_year = 2018 - int(new_df["Years"])
            if start_year < 1980:
                start_year = 1980
            last_year_smb = True
            dataset = GlacierDataset3D(name, start_year, 2018, last_year=last_year_smb, path="glacier_dmdt.csv")
            # dataset = GlacierDatasetNoPadding3D(name, start_year, 2018, last_year=last_year_smb, path="glacier_dmdt.csv")
            datasets = train_val_dataset(dataset, val_split=0.3, shuffle=False)
            last_year_dmdt = Glacier_dmdt(name, start_year - 1, 2017, path="glacier_dmdt.csv")
            train_loader = DataLoader(datasets['train'], batch_size=1)
            test_dataset = datasets['val']
            extractor = SeparateFeatureExtractor3D(output_dim=256, layers=[
                TWCNN2D(),TWCNN2D(),TWCNN2D(),TWCNN2D(),TWCNN2D(),TWCNN2D(), TWCNN2D()
            ])
            ann_model = ANNPredictor2(layers=None, input_dim=256, p=0.5, use_last_year_smb=last_year_smb)
            # ann_model = ANNPredictor2(layers=None, input_dim=256, hidden_dim=256, n_layers=1, bidirection=False, p=0.5)
            glacier_model = GlacierModel(extractor, ann_model, use_last_year_smb=last_year_smb, name="TWCNNLSTM3DNOPADDING")
            cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loss_function = torch.nn.MSELoss()
            trainer(glacier_model, train_loader=train_loader, testdataset=test_dataset, show=False,
                    test_last_year_dmdt=last_year_dmdt, use_last_year=last_year_smb,
                    device=cuda, epochs=20, lr=0.002, reg=0.001, save_every=10, eval_every=1, test_split_at=15,
                    critic=loss_function, optimizer=torch.optim.Adam, save_path="saved_models")
        except Exception as e:
            print(e)
            continue