import os
import torch
import numpy as np
import pandas as pd
import netCDF4 as nc
from os.path import join
from torch.utils.data import Dataset
from tqdm import tqdm

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
        return torch.cat(data, dim=0)


class Glacier_dmdt(Dataset):
    def __init__(self, glacier, start_year, end_year, path="glaicer_dmdt.csv"):
        super(Glacier_dmdt, self).__init__()
        self.df = pd.read_csv(path)
        if start_year > end_year:
            end_year, start_year = start_year, end_year
        self.glacier = glacier
        self.start_year = start_year
        self.end_year = end_year

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


class NewGlacierDataset(Dataset):
    def __init__(self, glacier_name, start_year, end_year, path="glaicer_dmdt.csv"):
        super(NewGlacierDataset, self).__init__()
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        self.start_year = start_year
        self.end_year = end_year
        self.glacier_name = glacier_name
        self.df = pd.read_csv(path)
        start_indx, end_idx = self.get_index_year()
        self.index_dict, self.new_df = self.get_index_dict()
        self.ERA5Data = [data[start_indx:end_idx + 1] for data in extract_data(glacier_name)]

    def get_index_year(self):
        if self.start_year < 1971 or self.start_year > 2017:
            raise ValueError(f"Start year does not exist: {self.start_year}")
        if self.end_year < 1972 or self.end_year > 2018:
            raise ValueError(f"End year does not exist: {self.end_year}")
        return self.start_year - 1971, self.end_year - 1972
    
    def get_index_dict(self):
        new_df = self.df[self.df["NAME"] == self.glacier_name]
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
        return self.end_year - self.start_year + 1

    def __getitem__(self, index):
        x = np.array([data[index] for data in self.ERA5Data])
        return x, float(self.new_df[self.index_dict[index]])


# TODO please fix error line: cond_1 = df_1["ALL_same"] == "FALSE" raise KeyError(key) from err KeyError: 'ALL_same'
def clean_glaicer_select(glacier_path="glaicer_dmdt.csv", glacier_path2="Glacier_select.csv"):
    df_1 = pd.read_csv(glacier_path)
    df_2 = pd.read_csv(glacier_path2)
    cond_1 = df_1["ALL_same"] == "FALSE"
    df2_filter = df_2[cond_1]
    df_1_year = df_1[["NAME", "Years"]][cond_1]
    df_1_year = df_1_year.set_index(df_1_year["NAME"])
    df_1_year = df_1_year.drop(columns=["NAME"])

    df2_filter = df2_filter.set_index(df2_filter["NAME"])
    df2_filter = df2_filter.drop(columns=["NAME"])
    print(df2_filter)
    print(df_1_year)
    df2_filter = pd.concat([df2_filter, df_1_year], axis=1)
    df2_filter.to_csv(glacier_path2)
    return df2_filter


def extract_data(name):
    # get all types of data
    file_obj = nc.Dataset(
        "./new_era5_data.nc")  # ['longitude', 'latitude', 'expver', 'time', 'u10', 'v10', 'd2m', 't2m', 'msl', 'mwd', 'sst', 'sp', 'tp']
    k = file_obj.variables.keys()
    temperature = file_obj.variables['t2m'][:]
    pressure = file_obj.variables['sp'][:]
    dew_point_temperature = file_obj.variables['d2m'][:]
    glaciers = pd.read_csv('./Glacier_select.csv')
    t = file_obj.variables['time']
    times = nc.num2date(t, 'hours since 1900-01-01 00:00:00', only_use_python_datetimes=True,
                        only_use_cftime_datetimes=False)[:]
    lon = file_obj.variables['longitude'][:]
    lat = file_obj.variables['latitude'][:]
    sst = file_obj.variables['sst'][:].filled(-999)
    tcc = file_obj.variables['tcc'][:]
    total_precipitation = file_obj.variables['tp'][:]
    u_wind = file_obj.variables['u10'][:]
    v_wind = file_obj.variables['v10'][:]
    temperature_data = np.empty((0, 15, 64), np.float64)
    pressure_data = np.empty((0, 15, 64), np.float64)
    wind_data = np.empty((0, 15, 64), np.float64)
    precipitation_data = np.empty((0, 15, 64), np.float64)
    cloudcover_data = np.empty((0, 15, 64), np.float64)
    ocean_data = np.empty((0, 15, 64), np.float64)
    dew_point_temperature_data = np.empty((0, 15, 64), np.float64)
    # proceed on each glacier
    for m in range(len(glaciers)):
        glacier_name = glaciers.loc[m]['NAME']
        if glacier_name != name:
            continue
        glat = glaciers.loc[m]['LAT']
        glon = glaciers.loc[m]['LON']
        area = glaciers.loc[m]['AREA']
        r = np.sqrt(area)
        left1 = len(lon)
        left2 = len(lon)
        right1 = 0
        down1 = 0
        right2 = 0
        down2 = 0
        up1 = len(lat)
        up2 = len(lat)
        # find 4 bounds of the square
        for i in range(len(lat)):
            for j in range(len(lon)):
                d = haversine(glon, glat, lon[j], lat[i])
                if d <= 200:
                    left1 = min(left1, j)
                    right1 = max(right1, j)
                    up1 = min(up1, i)
                    down1 = max(down1, i)
                if d <= r:
                    left2 = min(left2, j)
                    right2 = max(right2, j)
                    up2 = min(up2, i)
                    down2 = max(down2, i)
        # from every month
        print("Get data...")
        for tt in tqdm(range(4, len(times) - 35)):
            # get data
            ocean_data_temp = []
            temperature_data_temp = []
            wind_data_temp = []
            pressure_data_temp = []
            precipitation_data_temp = []
            cloudcover_data_temp = []
            dew_point_temperature_data_temp = []
            for i in range(up1, down1 + 1):
                temp1, temp2, temp3, temp4, temp5, temp6, temp7 = [], [], [], [], [], [], []
                for j in range(left1, right1 + 1):
                    # check if inside the glacier
                    if left2 <= j <= right2 and up2 <= i <= down2:
                        if sst[0][0][i][j] == -999:
                            temp1.append(temperature[tt][0][i][j])
                            temp2.append(np.sqrt(u_wind[tt][0][i][j] ** 2 + v_wind[tt][0][i][j] ** 2))
                            temp3.append(pressure[tt][0][i][j])
                            temp4.append(total_precipitation[tt][0][i][j])
                            temp5.append(tcc[tt][0][i][j])
                            temp6.append(dew_point_temperature[tt][0][i][j])
                        else:
                            temp1.append(0.0)
                            temp2.append(0.0)
                            temp3.append(0.0)
                            temp4.append(0.0)
                            temp5.append(0.0)
                            temp6.append(0.0)
                    else:
                        temp1.append(0.0)
                        temp2.append(0.0)
                        temp3.append(0.0)
                        temp4.append(0.0)
                        temp5.append(0.0)
                        temp6.append(0.0)
                    # check ocean data
                    if sst[0][0][i][j] != -999:
                        temp7.append(sst[tt][0][i][j])
                    else:
                        temp7.append(0.0)
                temperature_data_temp.append(temp1)
                wind_data_temp.append(temp2)
                pressure_data_temp.append(temp3)
                precipitation_data_temp.append(temp4)
                cloudcover_data_temp.append(temp5)
                dew_point_temperature_data_temp.append(temp6)
                ocean_data_temp.append(temp7)
            temperature_data_temp = np.array(temperature_data_temp)
            wind_data_temp = np.array(wind_data_temp)
            pressure_data_temp = np.array(pressure_data_temp)
            precipitation_data_temp = np.array(precipitation_data_temp)
            cloudcover_data_temp = np.array(cloudcover_data_temp)
            dew_point_temperature_data_temp = np.array(dew_point_temperature_data_temp)
            ocean_data_temp = np.array(ocean_data_temp)
            temperature_data = np.append(temperature_data, data_padding(temperature_data_temp), axis=0)
            wind_data = np.append(wind_data, data_padding(wind_data_temp), axis=0)
            pressure_data = np.append(pressure_data, data_padding(pressure_data_temp), axis=0)
            precipitation_data = np.append(precipitation_data, data_padding(precipitation_data_temp), axis=0)
            cloudcover_data = np.append(cloudcover_data, data_padding(cloudcover_data_temp), axis=0)
            dew_point_temperature_data = np.append(dew_point_temperature_data,
                                                   data_padding(dew_point_temperature_data_temp), axis=0)
            ocean_data = np.append(ocean_data, data_padding(ocean_data_temp), axis=0)
        break
    return temperature_data.reshape(-1, 12, 15,64), wind_data.reshape(-1, 12, 15,64), pressure_data.reshape(-1, 12, 15,64), precipitation_data.reshape(-1, 12, 15,64), cloudcover_data.reshape(-1, 12, 15,64), dew_point_temperature_data.reshape(-1, 12, 15,64), ocean_data.reshape(-1, 12, 15,64)


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


# padding data to fixed shape
def data_padding(data):
    fixed_matrix = np.zeros((15, 64), dtype=np.float64)
    fixed_matrix[0:data.shape[0], 0:data.shape[1]] = data
    res = np.expand_dims(fixed_matrix, 0)
    return res


