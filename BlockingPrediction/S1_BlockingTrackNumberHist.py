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
import imageio
from matplotlib.dates import DateFormatter

sys.stdout.reconfigure(line_buffering=True) # print at once in slurm

# get the LWA for the blocking and check the LWA for the track
# plot1: the map of the LWA
# plot2: the contour of the track's LWA
# plot3: the line of the LWA for both blocking and track

# %% 00 function preparation --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA", "ALL"]
seasonsmonths = [[12, 1, 2], [6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["CC", "AC"]

def Region_ERA(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 300, 60
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max = 40, 70, 130, 250
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max = -75, -45, 180, 300
    return lat_min, lat_max, lon_min, lon_max

def PlotBoundary(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 250, 90, 350
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 80, 280, 180
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = -90, -30, 130, 330, 230

    return lat_min, lat_max, lon_min, lon_max, loncenter

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

# %% 01 get the data ------------------------
typeid = 1
rgname = "ATL"
ss = "ALL"
cyc = "AC"

# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

ss = 'ALL'
interaction_data = {}

for typeid in [1,2,3]:
    for cyc in cycTypes:
        for rgname in regions:

            key = (rgname, typeid, cyc)

            # read in lat and lon for LWA
            lon = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
            lat = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order 90~-90
            # get the lat and lon for LWA, grouped by NH and SH
            lat_mid = int(len(lat)/2) + 1 
            if rgname == "SP":
                latLWA = lat[lat_mid:len(lat)]
            else:
                latLWA = lat[0:lat_mid-1]
            latLWA = np.flip(latLWA) # make it ascending order (from south to north)
            print(latLWA, flush=True)
            lonLWA = lon 

            # get the event's label
            if rgname == "SP":
                with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily_SH", "rb") as fp:
                    Blocking_diversity_label = pickle.load(fp)
                with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily_SH", "rb") as fp:
                    Blocking_diversity_date = pickle.load(fp)
            else:
                with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily", "rb") as fp:
                    Blocking_diversity_label = pickle.load(fp)
                with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
                    Blocking_diversity_date = pickle.load(fp)
            Blocking_diversity_label = Blocking_diversity_label[typeid-1] # get the blocking event list for the typeid
            Blocking_diversity_date = Blocking_diversity_date[typeid-1] # get the blocking event date for the typeid

            # read in the event's LWA list
            with open(f"/scratch/bell/hu1029/LGHW/BlockEventDailyLWAList_1979_2021_Type{typeid}_{rgname}_{ss}.pkl", "rb") as f:
                BlkeventLWA = pickle.load(f) 

            # the id of each blocking event (not all events! just the events within the target region)
            with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
                blkEindex = pickle.load(f)
            blkEindex = np.array(blkEindex)

            # EddyNumber = np.load(f'/scratch/bell/hu1029/LGHW/BlockingType{typeid}_EventEddyNumber_1979_2021_{rgname}_{ss}_{cyc}.npy')
            eddyBlockIndex = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')

            # get the enter/leaving time for each eddy [the location relative to the blocking]
            entertime = np.load(f'/scratch/bell/hu1029/LGHW/EnterTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
            leavetime = np.load(f'/scratch/bell/hu1029/LGHW/LeaveTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
                            
            # %% 02 test plotting for one event ------------------------
            interactingBlkID = eddyBlockIndex
            interactingBlkID = interactingBlkID[np.where(interactingBlkID >= 0)]
            unique_ids, counts = np.unique(interactingBlkID, return_counts=True) # counts is the number of eddies in each block (unique_ids)
            count_vals, count_freq = np.unique(counts, return_counts=True) # count_vals is the number of eddies in each block, count_freq is the frequency of each count value
            
            interaction_data[key] = {
                'count_vals': count_vals,  
                'count_freq': count_freq   
            }

            with open('blocking_eddy_counts_distribution.txt', 'a') as f:
                for c, freq in zip(count_vals, count_freq):
                    f.write(f"{rgname} N of type {typeid} that have {c} times {cyc} interaction: {freq}\n")
                    

            eventPersistence = np.load(f'/scratch/bell/hu1029/LGHW/BlockingEventPersistence_Type{typeid}_{rgname}_{ss}.npy')
            EddyNumber = np.load(f'/scratch/bell/hu1029/LGHW/BlockingType{typeid}_EventEddyNumber_1979_2021_{rgname}_{ss}_{cyc}.npy')
    
    with open('blocking_eddy_counts_distribution.txt', 'a') as f:    
        f.write('----------------------------------\n')

unique_pairs = {(k[0], k[1]) for k in interaction_data.keys()}
print(unique_pairs)

for rgname, typeid in unique_pairs:
    key_AC = (rgname, typeid, 'AC')
    key_CC = (rgname, typeid, 'CC')

    c_vals_AC = interaction_data[key_AC]['count_vals']
    freqs_AC = interaction_data[key_AC]['count_freq']
    c_vals_CC = interaction_data[key_CC]['count_vals']
    freqs_CC = interaction_data[key_CC]['count_freq']

    all_c_vals = np.union1d(c_vals_AC, c_vals_CC)
    print(all_c_vals, flush=True)

    freqs_AC_full = np.array([freqs_AC[np.where(c_vals_AC == c)[0][0]] if c in c_vals_AC else 0 for c in all_c_vals])
    freqs_CC_full = np.array([freqs_CC[np.where(c_vals_CC == c)[0][0]] if c in c_vals_CC else 0 for c in all_c_vals])

    bar_width = 0.4
    x = np.arange(len(all_c_vals))

    # plot
    plt.figure(figsize=(6, 4))
    plt.bar(x - bar_width/2, freqs_AC_full, width=bar_width, label='AC', color='blue', alpha=0.7)
    plt.bar(x + bar_width/2, freqs_CC_full, width=bar_width, label='CC', color='red', alpha=0.7)
    plt.xticks(x, all_c_vals)
    plt.xlabel('Number of eddy interactions per blocking')
    plt.ylabel('Frequency')
    plt.title(f'{rgname} Type {typeid} Eddy Interaction Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'NumberBar_type{typeid}_{rgname}_interaction_bar.png', dpi=300)
    plt.close()
    