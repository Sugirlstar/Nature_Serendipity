# get the density of the tracks during blocking / non-blocking days and plot the map

import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math

from netCDF4 import Dataset
import pandas as pd
from dateutil.relativedelta import relativedelta
import matplotlib.colors
import os
import cartopy
from cartopy import crs as ccrs
import matplotlib.ticker as mticker
import matplotlib.ticker as ticker
from scipy import ndimage
from multiprocessing import Pool, Manager
import cartopy.feature as cfeature
from scipy.ndimage import convolve
from scipy.signal import detrend
import pickle
import xarray as xr
import regionmask
from matplotlib.patches import Polygon
import matplotlib.path as mpath
from matplotlib.lines import Line2D
from multiprocessing import Pool, Manager
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns
import imageio
from scipy import stats
from collections import defaultdict

# 00 function --------------------------------
def findClosest(lati, latids):

    if isinstance(lati, np.ndarray):  # if lat is an array
        closest_indices = []
        for l in lati:  
            diff = np.abs(l - latids)
            closest_idx = np.argmin(diff) 
            closest_indices.append(closest_idx)
        return closest_indices
    else:
        # if lat is a single value
        diff = np.abs(lati - latids)
        return np.argmin(diff) 

# time management --------------------------------
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
timeiarr = np.array(timei)

# 01 read data --------------------------------------------------------------
# F128 attributes
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lonF128 = np.array(ds['lon'])
latF128 = np.array(ds['lat'])
latF128 = np.flip(latF128) # lat is in increasing order now (-90~90)
latF128_mid = findClosest(0,latF128)+1
latF128_NH = latF128[latF128_mid:len(latF128)] # increasing order 
lafF128_SH = latF128[0:latF128_mid] # increasing order
print('F128 NH:')
print(latF128_NH) # 0.35087653~89.46282157
print('F128 SH:')
print(lafF128_SH) # -90~-0.35087653

# 1dg attributes
ds1dg = xr.open_dataset("/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc")
lat1dg = np.array(ds1dg['lat']) # increasing order (-90~90)
lon1dg = np.array(ds1dg['lon'])
lat1dg_mid = int(len(lat1dg)/2) + 1 #91
lat1dgNH = lat1dg[lat1dg_mid:len(lat1dg)] # increasing order (1~90), exclude equator
lat1dgSH = lat1dg[0:lat1dg_mid-1] # increasing order (-90~0), exclude equator
print('1dg NH:')
print(lat1dgNH)
print('1dg SH:')
print(lat1dgSH)

# 02 prepare the functions for all the situations ------------------------------
def getTrackArr(trackType ,track_data, lonF128, latF128_NH, lon1dg, lat1dgNH, timeiarr):
   
    # extract all the point elements
    track_id_list = np.array([track_id for track_id, track in track_data for _ in track]) 
    time_list = np.array([time for _, track in track_data for (time, _, _) in track])
    y_list = np.array([y for _, track in track_data for (_, y, _) in track]) #lon values
    x_list = np.array([x for _, track in track_data for (_, _, x) in track]) #lat values

    print(f"{trackType} track total number:", len(time_list))

    # 1 - F128 array ------------------------
    # 01 make an array representing the trackpoint density
    trackPoints_array = np.zeros((len(timeiarr), len(latF128_NH), len(lonF128)))
    time_indices = np.searchsorted(timeiarr, time_list)
    lat_indices = np.array([findClosest(x, latF128_NH) for x in x_list])
    lon_indices = np.array([findClosest(y, lonF128) for y in y_list])
    # count numbers save in trackPoints_array
    np.add.at(trackPoints_array, (time_indices, lat_indices, lon_indices), 1)
    # 02 get the track ID array
    trackPoints_ID = np.zeros((len(datetime_array), len(latF128_NH), len(lonF128)))
    for t, la, lo, tid in zip(time_indices, lat_indices, lon_indices, track_id_list):
        trackPoints_ID[t, la, lo] = tid 
    trackPoints_array = trackPoints_array.astype(bool)
    
    # 2 - 1dg array ------------------------
    # 01 make an array representing the trackpoint density
    trackPoints_array_1dg = np.zeros((len(timeiarr), len(lat1dgNH), len(lon1dg)))
    time_indices_1dg = np.searchsorted(timeiarr, time_list)
    lat_indices_1dg = np.array([findClosest(x, lat1dgNH) for x in x_list])
    lon_indices_1dg = np.array([findClosest(y, lon1dg) for y in y_list])
    # count numbers save in trackPoints_array
    np.add.at(trackPoints_array_1dg, (time_indices_1dg, lat_indices_1dg, lon_indices_1dg), 1)
    # 02 get the track ID array
    trackPoints_ID_1dg = np.zeros((len(datetime_array), len(lat1dgNH), len(lon1dg)))
    for t, la, lo, tid in zip(time_indices_1dg, lat_indices_1dg, lon_indices_1dg, track_id_list):
        trackPoints_ID_1dg[t, la, lo] = tid
    trackPoints_array_1dg = trackPoints_array_1dg.astype(bool)

    return trackPoints_array, trackPoints_ID, trackPoints_array_1dg, trackPoints_ID_1dg

# 03 get AC/CC arrs ===========================
# tracks

for k in ['NH', 'SH']:
    for trackType in ['AC', 'CC']:

        with open(f'/scratch/bell/hu1029/LGHW/{trackType}Zanom_allyearTracks_{k}.pkl', 'rb') as file:
            track_data = pickle.load(file)

        if k == 'SH':
            lat128_k = lafF128_SH
            lat1dg_k = lat1dgSH
        else:
            lat128_k = latF128_NH
            lat1dg_k = lat1dgNH
            
        trackPoints_array_k, trackPoints_ID_k, trackPoints_array_1dg_k, trackPoints_ID_1dg_k = getTrackArr(trackType, track_data, lonF128, lat128_k, lon1dg, lat1dg_k, timeiarr)
        
        # 01 save the bool arrays - F128 and 1dg
        np.save(f'/scratch/bell/hu1029/LGHW/{trackType}trackPoints_arrayF128_{k}.npy', trackPoints_array_k)
        np.save(f'/scratch/bell/hu1029/LGHW/{trackType}trackPoints_array1dg_{k}.npy', trackPoints_array_1dg_k)
        
        trackPoints_frequency = np.nansum(trackPoints_array_k, axis=0) 
        # plot the map F128 -------------------
        if k == 'NH':
            lowlat = 20; highlat = 90
        else:
            lowlat = -90; highlat = -20
        fig, ax, cf = create_Map(lonF128,lat128_k,trackPoints_frequency,fill=True,fig=None,
                                    minv=0, maxv=np.nanmax(trackPoints_frequency), interv=11, figsize=(12,5),
                                    lowerlat=lowlat, upperlat=highlat,
                                    centralLon=0, colr='PuBu', extend='max',title=f'{trackType} tracks density')
        addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
        plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
        plt.show()
        plt.tight_layout()
        plt.savefig(f'{trackType}_pointFrequency_{k}_F128.png')
        plt.close()

        trackPoints_frequency_1dg = np.nansum(trackPoints_array_1dg_k, axis=0)
        # plot the map 1dg -------------------
        fig, ax, cf = create_Map(lon1dg,lat1dg_k,trackPoints_frequency_1dg,fill=True,fig=None,
                                    minv=0, maxv=np.nanmax(trackPoints_frequency_1dg), interv=11, figsize=(12,5),
                                    lowerlat=lowlat, upperlat=highlat,
                                    centralLon=0, colr='PuBu', extend='max',title=f'{trackType} tracks density')
        addSegments(ax,[(330,45),(30,45),(30,75),(330,75),(330,45)],colr='black',linewidth=2)
        plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
        plt.show()
        plt.tight_layout()
        plt.savefig(f'{trackType}_pointFrequency_{k}_1dg.png')
        plt.close()
        
        # 02 save the track ID array - F128 and 1dg
        np.save(f'/scratch/bell/hu1029/LGHW/{trackType}trackPoints_TrackIDarray_{k}_F128.npy', trackPoints_ID_k)
        np.save(f'/scratch/bell/hu1029/LGHW/{trackType}trackPoints_TrackIDarray_{k}_1dg.npy', trackPoints_ID_1dg_k)
