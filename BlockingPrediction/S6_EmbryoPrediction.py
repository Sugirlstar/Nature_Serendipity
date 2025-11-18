"""
============================================================
Script:      S6_EmbryoPrediction.py
Author:      Yanjun Hu
Created:     2025-06-10
Purpose:     

1. read in the embryo event index array and remove those not in the RealEmbryoIDs
2. read in the embryo events that successfully develop into Ridge/Trough blocking events
3. read in the AC and CC track data
4. find the related AC and CC tracks for each embryo (same as the eddy-blocking interaction)
5. calculate the probability of blocking events developing from an embryo with/without AC or CC eddies

Inputs:
    - embryoArr: 3D numpy array of integer embryo IDs
    - RealEmbryoIDs: 1D numpy array of valid embryo IDs

Outputs:
    - embryoArr (modified in place): all invalid IDs replaced with -1
    - Optional print: number of values replaced

Usage:
    - Directly run in a Python environment or Jupyter Notebook

Dependencies:
    - see the import statements below
============================================================
"""

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
import cv2
import copy
from collections import defaultdict
from scipy.interpolate import RegularGridInterpolator

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

def get_lon_width(finalLabel,nlon):

    _, x_idx = np.where(finalLabel > 0)
    x_sorted = np.sort(np.unique(x_idx))
    gaps = np.diff(np.concatenate((x_sorted, [x_sorted[0] + nlon])))
    max_gap_idx = np.argmax(gaps)
    west_idx = (x_sorted[(max_gap_idx + 1) % len(x_sorted)]) % nlon
    east_idx = (x_sorted[max_gap_idx]) % nlon
    if west_idx > east_idx:
        lon_width = 360 - west_idx + east_idx
    else:
        lon_width = east_idx - west_idx 
    return lon_width

def getSliceSingle(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, 
                latup=40, latdown=40, lonleft=80, lonright=40):
    
    LWAlatStart = latid-latdown if latid-latdown >= 0 else 0 
    LWAlatEnd = latid+latup if latid+latup <= len(latLWA) else len(latLWA)
    LWALatSlice = LWA_td[LWA_dayi, LWAlatStart:LWAlatEnd, :]

    if latid-latdown < 0:
        num_new_rows = abs(latid - latdown)
        new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
        LWALatSlice = np.vstack((new_rows, LWALatSlice))
        print(f'Added {num_new_rows} rows at the top. Now shape: {LWALatSlice.shape}', flush=True)

    if latid + latup > len(latLWA):
        num_new_rows = latid + latup - len(latLWA)
        new_rows = np.full((num_new_rows, LWALatSlice.shape[1]), np.nan)
        LWALatSlice = np.vstack((LWALatSlice, new_rows))
        print(f'Added {num_new_rows} rows at the bottom. Now shape: {LWALatSlice.shape}', flush=True)

    # lon
    start = lonid - lonleft
    end = lonid + lonright
    if start < 0:  
        indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
    elif end >= len(lonLWA):
        indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
    else:
        indices = list(range(start, end))
    slice_LWA = LWALatSlice[:,indices]

    return slice_LWA

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

# # %% 01 get the LWA events ------------------------
# # read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

ss = 'ALL'
rgname = "ATL"

dstrack = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
tracklon = np.array(ds['lon'])
tracklat = np.array(ds['lat'])
tracklat = np.flip(tracklat)
if rgname == "SP":
    tracklat = tracklat[0:(findClosest(0,tracklat)+1)]
else:
    tracklat = tracklat[(findClosest(0,tracklat)+1):len(tracklat)]

lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)
print(lat_min1, lat_max1, lon_min1, lon_max1)

if rgname == "SP":
    HMi = '_SH'
else:
    HMi = ''

# read in lat and lon for LWA
lon = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
lat = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order 90~-90
# get the lat and lon for LWA, grouped by NH and SH
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
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

# get the wave event
T = LWA_td.shape[0]
print('Total number of timesteps:', T, flush=True)

T4 = T // 4
LWA_Z = LWA_td[:T4*4]
LWA_Z = LWA_Z.reshape(T4, 4, len(latLWA), len(lonLWA)).mean(axis=1) # daily mean of LWA

print('LWA_Z loaded, shape: ', LWA_Z.shape, flush=True)


# %% read in necessary data for the composite
RealEmbryoIDs = np.load(f'/scratch/bell/hu1029/LGHW/embryo_RealEmbryoIDs_{rgname}.npy')

embryoArr = np.load(f'/scratch/bell/hu1029/LGHW/EmbryoIndexArrayDaily_{rgname}.npy') # the embryo event index array, daily value
# remove the embryoids not in the RealEmbryoIDs
mask = np.isin(embryoArr, RealEmbryoIDs)
embryoArr[~mask] = -1

# filter the embryo events to only keep the first three days of each embryo
id2locs = defaultdict(list)
Tt = embryoArr.shape[0]
# 只扫描一次数组，记录所有 (t, y, x, eid)
for t in range(Tt):
    ids_in_frame = np.unique(embryoArr[t])
    ids_in_frame = ids_in_frame[ids_in_frame != -1]
    for eid in ids_in_frame:
        ys, xs = np.where(embryoArr[t] == eid)
        for y, x in zip(ys, xs):
            id2locs[eid].append((t, y, x))

# 第二步：处理每个 embryo，只保留前三天的值
for eid, locs in id2locs.items():
    # 提取所有时间步
    times = [t for t, _, _ in locs]
    t_unique = sorted(set(times))
    
    if len(t_unique) > 3:
        # 允许保留的时间步
        t_keep = set(t_unique[:3])
        # 将不在前三天的像元设为 -1
        for t, y, x in locs:
            if t not in t_keep:
                embryoArr[t, y, x] = -1
        print(f'Embryo {eid} processed, kept {t_keep} time steps.', flush=True)

# get the IDs of the embryo events
nday = np.shape(embryoArr)[0] # number of days (not 6-hourly)
embryovalues = np.unique(embryoArr)  # get the unique embryo values
print('total embryo length:', len(embryovalues), flush=True)
embryovalues = embryovalues[embryovalues >= 0]  # remove the -1 values
print('embryo ids after removing invalid:', embryovalues, flush=True)
embryoArr = np.repeat(embryoArr, 4, axis=0)  # transfer to 6-hourly index

# %% check AC and CC probability ------------------------
# for each type of embryo, get the AC and CC related to it.
# for each related event, get the probability of if it successfully develops into a blocking event
SuccssEmbryoID = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccssEmbryoID_{rgname}.npy')  # the embryo IDs that successfully develop into a blocking event
all_included = np.all(np.isin(SuccssEmbryoID, RealEmbryoIDs))
if not all_included:
    print('Warning: Not all successful embryo IDs are included in the real embryo IDs.', flush=True)

# get the track lists
# read in the track's interaction information
if rgname == "SP":
    with open(f'/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks_SH.pkl', 'rb') as file:
        AC_track_data = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks_SH.pkl', 'rb') as file:
        CC_track_data = pickle.load(file)
else:
    with open(f'/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks.pkl', 'rb') as file:
        AC_track_data = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks.pkl', 'rb') as file:
        CC_track_data = pickle.load(file)
  
track_data = AC_track_data
BlockingEIndex = embryoArr
targetblockingeventID = embryovalues
# find the interacting AC and CC tracks for each embryo, similar with the blocking events
EddyNumber = [0] * len(targetblockingeventID) # a list of the eddy number that each block is related to; length = blocking event number
BlockIndex = [-1] * len(track_data) # a list of the blockingid that each track is related to; length = track number
tracknum = 0

for index, pointlist in track_data:

    tp = 'N'
    bkids = -1

    times = [ti for ti, _, _ in pointlist]

    # get the lat and lon id for blockings
    latids = [lati for _, _, lati in pointlist]
    latids = findClosest(latids,latLWA)
    lonids = [loni for _, loni, _ in pointlist]
    lonids = findClosest(lonids,lonLWA)
    timeids = [timei.index(i) for i in times]
    blockingValue = BlockingEIndex[np.array(timeids), np.array(latids), np.array(lonids)]
    blockingValue = blockingValue.astype(float) # convert to float for np.isnan check
    blockingValue[np.where(blockingValue == -1)] = np.nan  # replace -1 with NaN 
    # add a condition: only count the first three days of the embryo events

    # 001 Through (and Edge)
    if np.isnan(blockingValue[0]) and np.isnan(blockingValue[-1]):
        if np.any(blockingValue >= 0):
            indices = np.where(blockingValue >= 0)[0]
            if np.all(np.diff(indices) == 1):
                print(blockingValue,flush=True)
                bkEindex = np.unique(blockingValue) # the blockingid that the track is related to (global id)
                bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                bkids = bkEindex[0] # only get the first one that interacts with
                if len(bkEindex) > 1:
                    print(f'Warning: {len(bkEindex)} blocking events identified for track {index}', flush=True)
                for nn in bkEindex:
                    nnidx = np.where(targetblockingeventID == nn)[0][0] # the location of the event id in the region's collection
                    EddyNumber[nnidx] += 1
                print(f'{bkids}: Through Identified',flush=True)
            else:
                print(blockingValue,flush=True)
                bkEindex = np.unique(blockingValue) # the blockingid that the track is related to
                bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
                bkids = bkEindex[0] # only get the first one that interacts with
                for nn in bkEindex:
                    nnidx = np.where(targetblockingeventID == nn)[0][0]
                    EddyNumber[nnidx] += 1
                print(f'{bkids}: Edge Identified',flush=True)
    
    # 002 Absorbed
    if np.isnan(blockingValue[0]) and blockingValue[-1] >= 0:
        print(blockingValue,flush=True)
        bkEindex = np.unique(blockingValue)
        bkEindex = bkEindex[~np.isnan(bkEindex)].astype(int)
        print(bkEindex,flush=True)
        bkids = bkEindex[0]
        if len(bkEindex) > 1:
            print(f'Warning: {len(bkEindex)} blocking events identified for track {index}', flush=True)
        for nn in bkEindex:
            nnidx = np.where(targetblockingeventID == nn)[0][0]
            EddyNumber[nnidx] += 1
        print(f'{bkids}: Absorbed Identified',flush=True)

    # 003 Internal
    if blockingValue[0] >= 0:
        print(f'{bkids}: Internal or Spawned Identified',flush=True)

    BlockIndex[tracknum] = bkids
    tracknum += 1

# related to the AC track list
EddyNumber = np.array(EddyNumber)  # convert to numpy array
relatedACIDloc = np.where(EddyNumber > 0)[0]  # the global ids of the embryo that related to AC eddies
print('Related AC embryo ids locations:', relatedACIDloc, flush=True)
relatedACID = targetblockingeventID[relatedACIDloc]  # the global ids of the embryo that related to AC eddies
print('length of embryos with ACs:', len(relatedACID), flush=True)

# get the embryo related to Ridges and Troughs
ExcludedEmbryoID = [eid for eid in embryovalues if eid not in RealEmbryoIDs]
successNum = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType1_{rgname}.npy')
successNum[ExcludedEmbryoID] = 0 
RidgeSuccssEmbryo = np.where(successNum > 0)[0]  # the global ids of the embryo that successfully develop into a blocking event for type 1
print("RidgeEmbryo:", RidgeSuccssEmbryo, flush=True)
all_included = np.all(np.isin(RidgeSuccssEmbryo, RealEmbryoIDs))
if not all_included:
    print('Warning: Not all successful Ridge embryo IDs are included in the real embryo IDs.', flush=True)

successNum = np.load(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType2_{rgname}.npy')
successNum[ExcludedEmbryoID] = 0 
TroughSuccssEmbryo = np.where(successNum > 0)[0]  # the global ids of the embryo that successfully develop into a blocking event for type 1
print("DipoleEmbryo:", TroughSuccssEmbryo, flush=True)
all_included = np.all(np.isin(TroughSuccssEmbryo, RealEmbryoIDs))
if not all_included:
    print('Warning: Not all successful Trough embryo IDs are included in the real embryo IDs.', flush=True)

# calculate the probability of  Block | withEddy, Block | withoutEddy
print('total number of embryos:', len(RealEmbryoIDs), flush=True)
print(f'probability of blocking developed from an embryo: {len(SuccssEmbryoID)/len(RealEmbryoIDs)}', flush=True)
common_ridgeAC = np.intersect1d(RidgeSuccssEmbryo, relatedACID)
print(f'probability of Ridge blocking developed from an embryo  AC: {len(RidgeSuccssEmbryo)/len(RealEmbryoIDs)}', flush=True)
print(f'probability of Ridge blocking developed from an embryo with AC: {len(common_ridgeAC)/len(relatedACID)}', flush=True)
