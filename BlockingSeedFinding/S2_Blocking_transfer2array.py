import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math
import time

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
from scipy.ndimage import label
from scipy.interpolate import interp2d
import sys
import os

# %% function --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA", "ALL"]
seasonsmonths = [[12, 1, 2], [6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["CC", "AC"]
eventType = ["Blocking", "Seeding"]

def Region_ERA(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max = 45, 75, 300, 60
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max = 40, 70, 130, 250
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max = -75, -45, 180, 300
    return lat_min, lat_max, lon_min, lon_max

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

def getSectorValues(data, lat_min, lat_max, lon_min, lon_max, lat, lon):
    lat_indices = (lat >= lat_min) & (lat <= lat_max)
    lon_indices = (lon >= lon_min) & (lon <= lon_max)
    if lon_min > lon_max:
        lon_indices = (lon >= lon_min) | (lon <= lon_max)
    lat_indices = np.where(lat_indices)[0]
    lon_indices = np.where(lon_indices)[0]

    lat_indices, lon_indices = np.ix_(lat_indices, lon_indices)

    region_mask = np.zeros_like(data, dtype=bool)
    region_mask[:, lat_indices, lon_indices] = True

    data_filtered = np.where(region_mask, data, 0)

    return region_mask[0,:,:], data_filtered

# %% 01 prepare the data --------------------------------
# get lon and lat
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 
Blklon = lon

# time management 
Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
timestamp = list(Date0['date'])
timestamparr = np.array(timestamp)

# %% 02 filter the target region, season and transfer into 3d label array --------------------------------
for eve in eventType: 
    for rgname in regions:

        lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
        if eve == "Seeding":
            lon_min = lon_min - 30

        if rgname == 'SP':
            k = 'SH'
            Blklat = lat[lat_mid:len(lat)] # lat decreasing
        else:
            k = 'NH'
            Blklat = lat[0:lat_mid-1] # lat decreasing
            print(Blklat)

        with open(f"/scratch/bell/hu1029/LGHW/SD_{eve}_diversity_label_daily_{k}", "rb") as fp:
            Blocking_diversity_label = pickle.load(fp)   
        with open(f"/scratch/bell/hu1029/LGHW/SD_{eve}_diversity_date_daily_{k}", "rb") as fp:
            Blocking_diversity_date = pickle.load(fp)     
        with open(f"/scratch/bell/hu1029/LGHW/SD_{eve}_diversity_peaking_lon_daily_{k}", "rb") as fp:
            peakinglonList = pickle.load(fp)
        with open(f"/scratch/bell/hu1029/LGHW/SD_{eve}_diversity_peaking_lat_daily_{k}", "rb") as fp:
            peakinglatList = pickle.load(fp)

        for type_idx in range(3):
        
            for i in range(len(seasons)):

                ss = seasons[i]
                target_months = seasonsmonths[i] # the target months for each season
                print(ss)
                print(target_months)

                # Check each blocking event, to see if it's within the target region
                ATLlist = []
                print(type_idx,flush=True)
                print(f'Type{type_idx+1}, {rgname}, {ss}-------------filtering--------------',flush=True)
                blocking_array = np.zeros((len(timestamp), len(Blklat), len(Blklon)),dtype=np.bool_)
                blockingID_array = np.full((len(timestamp), len(Blklat), len(Blklon)),fill_value=-1, dtype=np.int32)

                for event_idx in range(len(Blocking_diversity_date[type_idx])):

                    peakinglat = peakinglatList[type_idx][event_idx]  # the peaking latitude
                    peakinglon = peakinglonList[type_idx][event_idx]  # the peaking longitude

                    event_dates = Blocking_diversity_date[type_idx][event_idx] # a list of dates

                    if all(date.month not in target_months for date in event_dates):
                        continue

                    timeindex = np.where(np.isin(timestamparr, event_dates))[0]  # find the time index in the total len
                    blklabelarr = np.array(Blocking_diversity_label[type_idx][event_idx]) # the 3d array of blocking label
                
                    # filter based on the peaking location
                    lat_in = lat_min <= peakinglat <= lat_max
                    if lon_min <= lon_max:
                        lon_in = lon_min <= peakinglon <= lon_max
                    else:
                        lon_in = (peakinglon >= lon_min) or (peakinglon <= lon_max)
                    if(lat_in and lon_in):
                        ATLlist.append(event_idx)
                        blocking_array[timeindex, :, :] = np.logical_or(blocking_array[timeindex, :, :],blklabelarr) # fill into the 3d array
                        t_idx, y_idx, x_idx = np.where(blklabelarr > 0)
                        abs_t_idx = timeindex[t_idx]
                        blockingID_array[abs_t_idx, y_idx, x_idx] = int(event_idx) # fill into the event id values
                        # print('Event_idx blocked in the target region: ',event_idx,flush=True)

                blocking_array = blocking_array.astype(bool)

                # save the blocking array 
                np.save(f"/scratch/bell/hu1029/LGHW/SD_{eve}FlagmaskClusters_Type{type_idx+1}_{rgname}_{ss}.npy", blocking_array)
                # save the blocking id list
                with open(f"/scratch/bell/hu1029/LGHW/SD_{eve}FlagmaskClustersEventList_Type{type_idx+1}_{rgname}_{ss}", "wb") as fp:
                    pickle.dump(ATLlist, fp)
                # save the id array
                np.save(f"/scratch/bell/hu1029/LGHW/SD_{eve}ClustersEventID_Type{type_idx+1}_{rgname}_{ss}.npy", blockingID_array)

                print('blockingarr saved ----------------',flush=True)

    print(f"{eve}_Type{type_idx+1}_{rgname}_{ss} All events filtered and saved", flush=True)


# %% 03 plot the map, for each, test
# calculate the sum over time -------------------
latglobe = lat[lat != 0]  # get the latitudes from 90 to 0
for eve in eventType:
    for type_idx in range(3):
        for i in range(len(seasons)):

            ss = seasons[i]
            rg1 = np.load(f"/scratch/bell/hu1029/LGHW/SD_{eve}FlagmaskClusters_Type{type_idx+1}_{regions[0]}_{ss}.npy")
            rg2 = np.load(f"/scratch/bell/hu1029/LGHW/SD_{eve}FlagmaskClusters_Type{type_idx+1}_{regions[1]}_{ss}.npy")
            rg3 = np.load(f"/scratch/bell/hu1029/LGHW/SD_{eve}FlagmaskClusters_Type{type_idx+1}_{regions[2]}_{ss}.npy")

            # test if there are overlaps
            intersection = (rg1 & rg2) 
            idt, idx, idy = np.where(intersection > 0)
            # test if there are intersections (one events that are considered twice in two regions)
            with open("Blocking_summary_logic2.txt", "a") as f:
                    f.write(f"Region1and2 intersection grids, Type{type_idx+1}_{ss}: {np.nansum(np.where(intersection > 0))}\n")

            rgNH = np.logical_or(rg1,rg2)
            rgworld = np.concatenate([rgNH, rg3], axis=1)

            # calculate the sum over time
            Blk1sum = np.nansum(rgworld, axis=0)

            # plot the map, seperated for types -------------------
            fig, ax, cf = create_Map(Blklon,latglobe,Blk1sum,fill=True,fig=None,leftlon=-180, rightlon=180, lowerlat=-90, upperlat=90,
                                        minv=0, maxv=round(np.nanmax(Blk1sum)), interv=11, figsize=(12,5),
                                        centralLon=270, colr='PuRd', extend='max',title=f'{eve} frequency')

            col = ['red','blue','green']
            for k,rgname in enumerate(regions):
                lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
                addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr=col[k],linewidth=2)
                addSegments(ax,[(lon_min-30,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min-30,lat_max),(lon_min-30,lat_min)],colr=col[k],linewidth=1)

            plt.colorbar(cf,ax=ax,orientation='horizontal',label='Frequency (days)',fraction=0.04, pad=0.1)
            plt.show()
            plt.tight_layout()
            plt.savefig(f'{eve}3RegionFrequency_{blkTypes[type_idx]}_{ss}.png')
            plt.close()

    print('Map plotted ----------------',flush=True)

