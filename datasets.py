import os
import torch
import pandas as pd
from os.path import join
from torch.utils.data import Dataset


class GlacierDataset(Dataset):
    def __init__(self, ERA5Data, dmdtData):
        if not isinstance(ERA5Data, list):
            ERA5Data = [ERA5Data]
        if not isinstance(dmdtData, list):
            dmdtData = [dmdtData]
        self.ERA5_datasets = {ERA5.glacier: ERA5 for ERA5 in ERA5Data}
        self.dmdt_datasets = {dmdt.glacier: dmdt for dmdt in dmdtData}
        self.ERA5_index = {}
        self.dmdt_index = {}
        self.indexs = []
        self.dataset_name = []
        for ERA5, dmdt in zip(ERA5Data, dmdtData):
            era5_start_index, era5_end_index, dmdt_start_index, dmdt_end_index, size = self.check_match(ERA5, dmdt)
            self.ERA5_index[ERA5.glacier] = (era5_start_index, era5_end_index)
            self.dmdt_index[dmdt.glacier] = (dmdt_start_index, dmdt_end_index)
            self.indexs.append(size)
            self.dataset_name.append(ERA5.glacier)
        self.data_size = sum(self.indexs)

    @staticmethod
    def check_match(ERA5, dmdt):
        if not (ERA5.glacier == dmdt.glacier):
            raise ValueError("Glacier name does not match ERA5: {} dmdt: {}".format(ERA5.glacier, dmdt.glacier))
        start_year = max(ERA5.start_year, dmdt.start_year)
        end_year = min(ERA5.end_year, dmdt.end_year)
        if end_year - start_year < 0:
            raise ValueError("Year does not match. Glacier: {}".format(ERA5.glacier))
        if not ((start_year == ERA5.start_year) and (start_year == dmdt.start_year)):
            print("[Warning] start year does not match, will use {} as start year {}".format(start_year, ERA5.glacier))
        if not ((end_year == ERA5.end_year) and (end_year == dmdt.end_year)):
            print("[Warning] End year does not match, will use {} as end year {}".format(end_year, ERA5.glacier))
        era5_start_index = ERA5.start_year - start_year
        dmdt_start_index = dmdt.start_year - start_year
        era5_end_index = ERA5.end_year - start_year
        dmdt_end_index = dmdt.end_year - start_year
        size = end_year - start_year
        return era5_start_index, era5_end_index, dmdt_start_index, dmdt_end_index, size

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        num, i = index, 0
        for v, count in enumerate(self.indexs):
            if num >= count:
                num -= count
                i = v
            else:
                i = v
                break
        glacier_name = self.dataset_name[i]
        ERA5dataset = self.ERA5_datasets[glacier_name]
        dmdtdataset = self.dmdt_datasets[glacier_name]
        era5_start_index, _ = self.ERA5_index[glacier_name]
        dmdt_start_index, _ = self.dmdt_index[glacier_name]
        return ERA5dataset[era5_start_index + num], dmdtdataset[dmdt_start_index + num]


class ERA5Datasets(Dataset):
    def __init__(self, glacier, start_year, end_year, start_month=5, path="ECMWF_reanalysis_data"):
        super(ERA5Datasets, self).__init__()
        if start_year > end_year:
            end_year, start_year = start_year, end_year
        self.glacier = glacier
        self.ERA5_path = "{}/{}/".format(path, glacier)
        self.start_year = start_year
        self.end_year = end_year
        self.start_month = start_month
        self.cloud_path = None
        self.precipitation_path = None
        self.pressure_path = None
        self.wind_path = None
        self.temp_path = None
        self.index_dict = {}
        all_path = os.listdir(self.ERA5_path)
        index = 0
        for year in range(start_year, end_year):
            keys = []
            for month in range(12):
                keys.append(self.get_year(year, month))
            self.index_dict[index] = keys
            index += 1
        for path in all_path:
            if "cloud" in path:
                self.cloud_path = join(self.ERA5_path, path)
            elif "precipitation" in path:
                self.precipitation_path = join(self.ERA5_path, path)
            elif "pressure" in path:
                self.pressure_path = join(self.ERA5_path, path)
            elif "wind" in path:
                self.wind_path = join(self.ERA5_path, path)
            elif "temp" in path:
                self.temp_path = join(self.ERA5_path, path)
        self.paths = [self.cloud_path, self.precipitation_path, self.pressure_path, self.wind_path, self.temp_path]

    def get_year(self, year, month):
        if (month + self.start_month) // 13 == 1:
            return "{}_{}".format(year + 1, (month + self.start_month) % 12)
        return "{}_{}".format(year, (month + self.start_month))

    @staticmethod
    def get_data(path, columns):
        df = pd.read_csv(path)
        return df[columns]

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, index):
        months = self.index_dict[index]
        data = []
        for path in self.paths:
            entry = self.get_data(path, months).to_numpy()
            data.append(torch.from_numpy(entry).unsqueeze(0).float())
        return data


class Glacier_dmdt(Dataset):
    def __init__(self, glacier, start_year, end_year, path="glaicer_dmdt.csv"):
        super(Glacier_dmdt, self).__init__()
        if start_year > end_year:
            end_year, start_year = start_year, end_year
        self.glacier = glacier
        self.start_year = start_year
        self.end_year = end_year
        self.df = pd.read_csv(path)
        self.index_dict, self.new_df = self.get_index_dict()

    def get_index_dict(self):
        new_df = self.df[self.df["NAME"] == self.glacier]
        index_dict = []
        start = False
        for year in self.df.columns:
            if start:
                index_dict.append(year)
            if str(self.start_year) in year:
                start = True
            if str(self.end_year) in year:
                break
        return index_dict, new_df

    def __len__(self):
        return len(self.index_dict)

    def __getitem__(self, index):
        return float(self.new_df[self.index_dict[index]])
