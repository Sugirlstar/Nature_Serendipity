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

# %% 00 function --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = [ "ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["AC", "CC"]

def Region_ERA(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 300, 60
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max = 40, 70, 130, 250
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max = -75, -45, 180, 300
    return lat_min, lat_max, lon_min, lon_max

def findClosest(lati, latids):

    if isinstance(lati, (np.ndarray, list)):  # if lat is an array or list
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

# ACtracks ===========================
# tracks
for ss in seasons:        
    for cyc in cycTypes:
        for typeid in [1, 2, 3]:
            for rgname in regions:
                
                print(f'Start: {cyc}, type {typeid}, {rgname}, {ss}', flush=True)

                # check if the result file exists
                if os.path.exists(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_{rgname}_{ss}.npy'):
                    print(f'File already exists: {cyc}trackInteracting_array_Type{typeid}_{rgname}_{ss}.npy', flush=True)
                    continue

                if rgname == "SP":
                    HMi = '_SH'
                else:
                    HMi = '_NH'

                # 01 read data --------------------------------------------------------------
                # attributes for tracks
                ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
                timesarr = np.array(ds['time'])
                datetime_array = pd.to_datetime(timesarr)
                timeiarr = list(datetime_array)
                
                # attributes for z500 (1dg) - track points must be put into the same grid as the 1dg z500
                lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
                lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
                lat_mid = int(len(lat)/2) + 1 
                if rgname == "SP":
                    Blklat = lat[lat_mid:len(lat)]
                else:
                    Blklat = lat[0:lat_mid-1]
                Blklat = np.flip(Blklat) # make it ascending order (from south to north)
                print(Blklat, flush=True)
                Blklon = lon 

                timesarr = np.array(ds['time'])
                datetime_array = pd.to_datetime(timesarr)
                timeiarr = list(datetime_array)

                # get the lat/lon range
                lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
                
                # get the interact id
                # find if the file exists
                if not os.path.exists(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy'):
                    print(f'File not found: TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy', flush=True)
                    continue
                InterTypeSec2CC = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')
                interidCC = np.where(InterTypeSec2CC != -1)[0] # the location of the tracks that intersect with the blocking

                # AC tracks
                with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks{HMi}.pkl', 'rb') as file:
                    track_data = pickle.load(file)
                track_data = [track_data[i] for i in interidCC] # get the interacting track id

                # extract all the point elements
                trackid = np.array([i for i, _ in track_data])
                trackidarr = np.repeat(trackid, [len(track) for _, track in track_data])
                time_list = np.array([time for _, track in track_data for (time, _, _) in track])
                y_list = np.array([y for _, track in track_data for (_, y, _) in track]) #lon
                x_list = np.array([x for _, track in track_data for (_, _, x) in track]) #lat

                # make an array representing the trackpoint density
                trackPoints_idarr = np.full((len(datetime_array), len(Blklat), len(Blklon)),fill_value=-1, dtype=np.int64)
                trackPoints_array = np.zeros((len(datetime_array), len(Blklat), len(Blklon)), dtype=np.int16)

                time_indices = np.searchsorted(timeiarr, time_list)
                lat_indices = np.array([findClosest(x, Blklat) for x in x_list])
                lon_indices = np.array([findClosest(y, Blklon) for y in y_list])
                print('check:',flush=True)
                print(f'{trackidarr[:50]}', flush=True)
                print(f'{time_indices[:50]}', flush=True)
                print(f'{lat_indices[:50]}', flush=True)
                print(f'{lon_indices[:50]}', flush=True)
                # count numbers save in trackPoints_array
                np.add.at(trackPoints_array, (time_indices, lat_indices, lon_indices), 1)
                trackPoints_idarr[time_indices, lat_indices, lon_indices] = trackidarr

                trackPoints_array.astype(bool)

                np.save(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_array_Type{typeid}_{rgname}_{ss}.npy', trackPoints_array)
                np.save(f'/scratch/bell/hu1029/LGHW/{cyc}trackInteracting_idarr_Type{typeid}_{rgname}_{ss}.npy', trackPoints_idarr)

                trackPoints_frequency = np.nansum(trackPoints_array, axis=0) 
                # plot the map -------------------
                fig, ax, cf = create_Map(Blklon,Blklat,trackPoints_frequency,fill=True,fig=None,leftlon=-180, rightlon=180, lowerlat=-90, upperlat=90,
                                            minv=0, maxv=np.nanmax(trackPoints_frequency), interv=11, figsize=(12,5),
                                            centralLon=270, colr='PuBu', extend='max',title=f'{cyc} tracks density')
                addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='darkred',linewidth=2)
                plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (times)',fraction=0.04, pad=0.1)

                plt.show()
                plt.tight_layout()
                plt.savefig(f'SD_{cyc}InteractingFrequency_Type{typeid}_{rgname}_{ss}.png')
                plt.close()

                print(f'Finished {cyc} interacting tracks density for type {typeid} in {rgname} during {ss}', flush=True)
