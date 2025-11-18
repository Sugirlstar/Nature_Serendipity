import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *

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
import sys
sys.stdout.reconfigure(line_buffering=True) # print at once in slurm

# %% 00 function preparation --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = ["ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["AC", "CC"]
HMs = ["_NH","_SH"]

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

def getTrackLWA(track_data, latLWA, lonLWA, LWA_td, timei):
    
    LWAtrack_Sec2 = []
    for index, pointlist in track_data:
        
        print(index,flush=True)

        times = [ti for ti, _, _ in pointlist]
        latids = [lati for _, _, lati in pointlist]
        lonids = [loni for _, loni, _ in pointlist]
        timeids = [timei.index(i) for i in times]

        latinLWA = findClosest(np.array(latids), latLWA)
        loninLWA = findClosest(np.array(lonids), lonLWA)

        eachTrackLWA=[]
        for j in range(len(latinLWA)):

            lat_idx = latinLWA[j]
            lon_idx = loninLWA[j]
            timeid = timeids[j]
            radius = 5
            # get the range
            lat_range = slice(max(lat_idx - radius, 0), min(lat_idx + radius + 1, LWA_td.shape[1]))
            lon_range = [(lon_idx - i) % 360 for i in range(radius, -radius-1, -1)]

            # extracted
            extracted_data = LWA_td[timeid, lat_range, lon_range]
            lwa_sum = np.nansum(extracted_data)
            eachTrackLWA.append(lwa_sum) # a list of each day's LWA

        LWAtrack_Sec2.append(eachTrackLWA)

    return LWAtrack_Sec2

# 01 load the data -------------------------------------------------------------
# track's time
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in LWA
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # lat decreasing
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 
print('-------- LWA loaded --------', flush=True)
# read in lat and lon
lon = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
lat = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order 90~-90

# Groups

for HMi in HMs:
    for cyc in cycTypes:

        # get the lat and lon for LWA, grouped by NH and SH
        lat_mid = int(len(lat)/2) + 1 
        if HMi == "_SH":
            latLWA = lat[lat_mid:len(lat)]
            LWA_td = LWA_td_origin[:,lat_mid:len(lat),:] 
        else:
            latLWA = lat[0:lat_mid-1]
            LWA_td = LWA_td_origin[:,0:lat_mid-1,:]
        latLWA = np.flip(latLWA) # make it ascending order (from south to north)
        LWA_td = np.flip(LWA_td, axis=1) 
        print(latLWA, flush=True)
        print('LWA shape: ', LWA_td.shape, flush=True)
        lonLWA = lon 

        # %% 02 calculate the tracks'LWA -------------------------------------------------------------
        with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks{HMi}.pkl', 'rb') as file:
            track_data = pickle.load(file)

        LWAtrack = getTrackLWA(track_data, latLWA, lonLWA, LWA_td, timei)

        with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021{HMi}.pkl', 'wb') as file:
            pickle.dump(LWAtrack, file)

        # plot the pdf of CC and AC LWA
        flat_LWAtrack = np.array([np.nanmean(sublist) for sublist in LWAtrack if len(sublist) > 0])

        plt.figure()
        sns.kdeplot(flat_LWAtrack, fill=True)
        plt.title(f'Alltracks_{cyc}tracksLWA_PDF{HMi}')
        plt.xlabel('TRACK LWA')
        plt.ylabel('Density')
        plt.show()
        plt.savefig(f'Alltracks_{cyc}tracksLWA_PDF{HMi}.png')
        
    print(f'Finished {cyc} tracks LWA for {HMi}', flush=True)


