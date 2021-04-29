import numpy as np
import netCDF4 as nc
import os
import pandas as pd
from torch.utils.data import Dataset
from torch import nn
from extract_data import extract_data

class ERA5Dataset(Dataset):
    def __init__(self, glacier_name, start_year, end_year):
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        self.start_year = start_year
        self.end_year = end_year
        self.glacier_name = glacier_name
        start_indx, end_idx = self.get_index_year()
        self.ERA5Data = [data[start_indx:end_idx+1] for data in extract_data(glacier_name)]

    
    def get_index_year(self):
        if self.start_year < 1971 or self.start_year > 2019:
            raise ValueError(f"Start year does not exist: {self.start_year}")
        if self.end_year <1971 or self.end_year > 2019:
            raise ValueError(f"End year does not exist: {self.end_year}")
        return (self.start_year-1971, self.end_year-1971)
    
    def __len__(self):
        return self.end_year-self.start+1 

    def __getitem__(self, index):
        x1, x2, x3, x4, x5, x6, x7 = [data[index] for data in self.ERA5Data]
        return x1, x2, x3, x4, x5, x6, x7
