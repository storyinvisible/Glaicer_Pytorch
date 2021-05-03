import os
import torch
import argparse
from datasets import ERA5Datasets, Glacier_dmdt

parser = argparse.ArgumentParser(description='Glacier SMB prediction')
parser.add_argument('-model', type=str, help='Path to the saved_model')
parser.add_argument('-data', type=str, help='Path to the dataset', default="ECMWF_reanalysis_data")
parser.add_argument('-glacier', type=str, help='Glacier name')
parser.add_argument('-year', type=int, help='Glacier name')
args = parser.parse_args()

if args.year < 1980 or args.year > 2018:
    print("[ERROR] Year not within the available range")
if not os.path.exists(args.model):
    print("[ERROR] Model file not exist")
model = torch.load(args.model)
dataset = ERA5Datasets(args.glacier, 1980, 2018, path=args.data)
smb = Glacier_dmdt(args.glacier, 1980, 2018, path="glacier_dmdt.csv")
result = model(dataset[args.year - 1980].unsqueeze(0))
print("[INFO] Predicted SMB for {} in {} is: {}".format(args.glacier, args.year, result.item()))
print("[INFO]    Actual SMB for {} in {} is: {}".format(args.glacier, args.year, smb[args.year - 1980]))
