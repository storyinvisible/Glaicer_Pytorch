import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import Glacier_dmdt, ERA5Datasets, GlacierDataset
from models import GlacierModel, ANNPredictor, LSTMPredictor, Predictor, HCNN, VCNN


def trainer(extractor, predictor, train_loader, test_loader, loss_func, optimizer, epochs=500, lr=0.002, reg=0.001,
            save_every=10, print_every=10, save_path=None, device=None):
    model = GlacierModel(extractor, predictor).to(device)
    critic = loss_func()
    optim = optimizer(model.parameters(), lr=lr, weight_decay=reg)
    step = 0
    for epoch in range(epochs):
        train_loss = 0
        for feature, target in train_loader:
            feature, target = Variable(feature).to(device), Variable(target).to(device)
            step += 1
            pred = model(feature)
            loss = critic(pred.squeeze(1), target.float())
            optim.zero_grad()
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optim.step()
            train_loss += loss.item()
            if step % print_every == 0:
                print("[INFO] Epoch {}|{}|{} Loss :{:.4f}".format(step, epoch, epochs, train_loss / print_every))
            if step % save_every == 0:
                torch.save(model, save_path)
        test_loss = 0
        with torch.no_grad():
            for feature, target in test_loader:
                feature, target = feature.to(device), target.to(device)
                pred = model(feature)
                loss = critic(pred.squeeze(1), target.float().to(device))
                test_loss += loss.item()
        print("[INFO] Epoch {}|{} Loss :{:.4f}".format(epoch, epochs, train_loss / print_every))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smb = Glacier_dmdt("JAKOBSHAVN_ISBRAE", 1980, 2002, path="glaicer_dmdt.csv")
    data = ERA5Datasets("JAKOBSHAVN_ISBRAE", 1980, 2002, path="ECMWF_reanalysis_data")
    dataset = GlacierDataset([data], [smb])
    train_loader = DataLoader(dataset, batch_size=16)
    test_loader = train_loader
    trainer(HCNN(in_channel=5, output_dim=256, vertical_dim=289, device=device),
            LSTMPredictor(layers=None, input_dim=256, hidden_dim=256, n_layers=1,
                          bi_direction=False, p=0.5),
            train_loader=train_loader, test_loader=test_loader,
            loss_func=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            device=device, epochs=500, lr=0.002, reg=0.001, save_every=10, print_every=10,
            save_path="saved_models/HCNN_LSTM_model.h5")
