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

# 01 read data --------------------------------------------------------------
# attributes
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
latNH = lat[(findClosest(0,lat)+1):len(lat)]
print(latNH)

timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
timeiarr = np.array(timei)

# ACtracks ===========================
# tracks
with open('/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks.pkl', 'rb') as file:
    track_data = pickle.load(file)

# extract all the point elements
track_id_list = np.array([track_id for track_id, track in track_data for _ in track]) # track id
time_list = np.array([time for _, track in track_data for (time, _, _) in track])
y_list = np.array([y for _, track in track_data for (_, y, _) in track]) #lon
x_list = np.array([x for _, track in track_data for (_, _, x) in track]) #lat

time_indices = np.searchsorted(timeiarr, time_list)
lat_indices = np.array([findClosest(x, latNH) for x in x_list])
lon_indices = np.array([findClosest(y, lon) for y in y_list])

print(len(time_list))
# make an array representing the trackpoint density
trackPoints_array = np.zeros((len(datetime_array), len(latNH), len(lon)))
for t, la, lo, tid in zip(time_indices, lat_indices, lon_indices, track_id_list):
    trackPoints_array[t, la, lo] = tid

np.save('/scratch/bell/hu1029/LGHW/ACtrackPoints_TrackIDarray.npy', trackPoints_array)

# CCtracks ===========================
# tracks
with open('/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks.pkl', 'rb') as file:
    track_data = pickle.load(file)

# extract all the point elements
track_id_list = np.array([track_id for track_id, track in track_data for _ in track]) # track id
time_list = np.array([time for _, track in track_data for (time, _, _) in track])
y_list = np.array([y for _, track in track_data for (_, y, _) in track]) #lon
x_list = np.array([x for _, track in track_data for (_, _, x) in track]) #lat

time_indices = np.searchsorted(timeiarr, time_list)
lat_indices = np.array([findClosest(x, latNH) for x in x_list])
lon_indices = np.array([findClosest(y, lon) for y in y_list])

print(len(time_list))
# make an array representing the trackpoint density
trackPoints_array = np.zeros((len(datetime_array), len(latNH), len(lon)))
for t, la, lo, tid in zip(time_indices, lat_indices, lon_indices, track_id_list):
    trackPoints_array[t, la, lo] = tid

np.save('/scratch/bell/hu1029/LGHW/CCtrackPoints_TrackIDarray.npy', trackPoints_array)

