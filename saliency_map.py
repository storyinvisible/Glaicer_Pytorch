import torch
import torch.nn as nn
from torchvision import models
from datasets import extract_data, GlacierDataset3D
import pandas as pd
from torch.utils.data import DataLoader
from models import SeparateFeatureExtractor3D, TWCNN2D, LSTMPredictor, GlacierModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
glacier_info = pd.read_csv("Glacier_select.csv")
glaciers =list(glacier_info["NAME"])
new_df = glacier_info[glacier_info["NAME"]=="STORE_GLETSCHER"]
start_year = 2018 - int(new_df["Years"])
if start_year < 1979:
    start_year = 1979
dataset = GlacierDataset3D("STORE_GLETSCHER", start_year, 2018, path="glacier_dmdt.csv")

# Initialize the model
model = torch.load('saved_models/TWCNNLSTM2D/TWCNNLSTM2D_model.h5')

# Set the model to run on the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set the model on Eval Mode
model.train()
loss_function = torch.nn.MSELoss()
optim = torch.optim.Adam
count = 0
for d, t in dataset:
    if count<15:
        count += 1
        continue
    d = torch.tensor(d).unsqueeze(0).type(torch.FloatTensor).to(device)
    d.requires_grad_()
    output = model(d)
    loss = loss_function(output.squeeze(1), torch.tensor([t]).type(torch.FloatTensor).to(device))
    loss.backward()
    saliency = d.grad.data.abs()[0, :, :, :, :]
    # saliency = saliency[0][0]
    # print(saliency.shape)
    for j in range(12):
        fig, ax = plt.subplots(7, 1)
        for i in range(7):
            ax[i].imshow(saliency[i][j].cpu(), cmap='hot')
            ax[i].axis('off')
        plt.tight_layout()
        plt.savefig("plots/attention_month_"+str(j))
        plt.close()
    break
