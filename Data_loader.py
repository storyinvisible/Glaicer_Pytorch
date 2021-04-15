from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import os
class ERA_5_datasets(Dataset):
    def __init__(self, Glacier,start_year,end_year, start_month=5):
        ERA5_path="ECMWF_reanalysis_data/{}/".format(Glacier)
        all_path=os.listdir(ERA5_path)
        self.cloud_path=None
        self.precipitation_path=None
        self.pressure_path=None
        self.wind_path=None
        self.temp_path=None

        self.index_dict={}
        index=0
        get_year = lambda year ,month : "{}_{}".format(year+1,(month+5)%12) if (month+5)//13 ==1 else "{}_{}".format(year,(month+5))
        for year in range(start_year, end_year,1):
            keys=[]
            for month in range(12):

                keys.append(get_year(year,month))

            self.index_dict[index]=keys
            index+=1
        print(self.index_dict)
        for path in all_path:
            if "cloud" in path:
                self.cloud_path= ERA5_path+path

            elif "precipitation" in path:
                self.precipitation_path=  ERA5_path+path
            elif "pressure" in path :
                self.pressure_path = ERA5_path+path
            elif "wind" in path:
                self.wind_path=ERA5_path+path
            elif "temp" in path :
                self.temp_path=ERA5_path+path
        self.paths=[ self.cloud_path,self.precipitation_path,self.pressure_path,self.wind_path,self.temp_path]
    def get_data(self, path,columns):
        df= pd.read_csv(path)
        return df[columns]
    def __len__(self):
        return len(self.index_dict)
    def __getitem__(self, index):
        months =self.index_dict[index]
        data=None
        for path in self.paths:
            entry=self.get_data(path,months).to_numpy()

            if isinstance(data,np.ndarray):
                data= np.concatenate((data,entry),axis=1)
            else:
                data = entry
        print(data.shape)
        return torch.tensor(data)

class Glacier_dmdt(Dataset):
    def __init__(self, Glacier, start_year, end_year):
        self.glacier=Glacier
        self.start_year=start_year
        self.end_year=end_year
        self.df= pd.read_csv("glaicer_dmdt.csv")
        self.index_dict , self.new_df =self.get_index_dict()
    def get_index_dict(self):
        condition = self.df["NAME"] == self.glacier
        new_df = self.df[condition]
        names = self.df.columns
        index_dict = []
        start=False
        for year in names:
            if str(self.start_year) in year:
                start=True
            if start :
                index_dict.append(year)
            if str(self.end_year) in year:
                break
        print(index_dict)
        return  index_dict, new_df
    def __len__(self):
        return len(self.index_dict)
    def __getitem__(self, index):
        dmdt= self.new_df[self.index_dict[index]]
        return float(dmdt)



# ERA5 =ERA_5_datasets("JAKOBSHAVN_ISBRAE",1980,2002)
# glacier_dmdt= Glacier_dmdt("JAKOBSHAVN_ISBRAE",1980,2002)
# print(glacier_dmdt[1])
