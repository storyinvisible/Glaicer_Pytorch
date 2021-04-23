import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_data(name):
    # get all types of data
    file_obj = nc.Dataset("new_era5_data/new_era5_data.nc")  # ['longitude', 'latitude', 'expver', 'time', 'u10', 'v10', 'd2m', 't2m', 'msl', 'mwd', 'sst', 'sp', 'tp']
    k = file_obj.variables.keys()
    temperature = file_obj.variables['t2m'][:]
    pressure = file_obj.variables['sp'][:]
    dew_point_temperature = file_obj.variables['d2m'][:]
    glaciers = pd.read_csv('./Glacier_select.csv')
    t = file_obj.variables['time']
    times = nc.num2date(t, 'hours since 1900-01-01 00:00:00', only_use_python_datetimes=True, only_use_cftime_datetimes=False)[:]
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
        if glacier_name!=name:
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
        for tt in range(len(times)-15):
            # get data
            print("data in ", times[tt])
            ocean_data_temp = []
            temperature_data_temp = []
            wind_data_temp = []
            pressure_data_temp = []
            precipitation_data_temp = []
            cloudcover_data_temp = []
            dew_point_temperature_data_temp = []
            for i in range(up1, down1+1):
                temp1, temp2, temp3, temp4, temp5, temp6, temp7 = [], [], [], [], [], [], []
                for j in range(left1, right1+1):
                    # check if inside the glacier
                    if left2 <= j <= right2 and up2 <= i <= down2:
                        if sst[0][0][i][j]==-999:
                            temp1.append(temperature[tt][0][i][j])
                            temp2.append(np.sqrt(u_wind[tt][0][i][j]**2 + v_wind[tt][0][i][j]**2))
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
                    if sst[0][0][i][j]!=-999:
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
            dew_point_temperature_data = np.append(dew_point_temperature_data, data_padding(dew_point_temperature_data_temp), axis=0)
            ocean_data = np.append(ocean_data, data_padding(ocean_data_temp), axis=0)
        break  
    return temperature_data, wind_data, pressure_data, precipitation_data, cloudcover_data, dew_point_temperature_data, ocean_data

def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371
    return c * r

# padding data to fixed shape
def data_padding(data):
    fixed_matrix = np.zeros((15, 64), dtype=np.float64)
    fixed_matrix[0:data.shape[0], 0:data.shape[1]] = data
    res = np.expand_dims(fixed_matrix, 0)
    return res
