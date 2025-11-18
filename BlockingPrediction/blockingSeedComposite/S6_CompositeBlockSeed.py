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


# %% 01 parameters setting -------------------
typeid = 1
rgname = "ATL"
ss = "ALL"
cyc = "AC"

lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)
print(lat_min1, lat_max1, lon_min1, lon_max1)


# %% 02 read in LWA, Z500, track data -------------------
# tracklon/tracklat (Gaussian grid)
# latLWA/lonLWA
# timei: 6-hourly time index

LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

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
print('LWA shape: ', LWA_td.shape, flush=True)
lonLWA = lon 
print('latLWA:', latLWA, flush=True)
print('lonLWA:', lonLWA, flush=True)

dstrack = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(dstrack['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
tracklon = np.array(dstrack['lon'])
tracklat = np.array(dstrack['lat'])
tracklat = np.flip(tracklat)
if rgname == "SP":
    tracklat = tracklat[0:(findClosest(0,tracklat)+1)]
else:
    tracklat = tracklat[(findClosest(0,tracklat)+1):len(tracklat)]
print('tracklat:', tracklat, flush=True)
print('tracklon:', tracklon, flush=True)

# 6-hourly Blocking event id array
blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
# transfer to 6-hourly
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

# get the blocking index each track contribute to (-1 or the blocking global index), 1d, same length as the track_data
InteractingBlockID = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy') 
InteractingBlockID_positive = np.where(InteractingBlockID != -1)[0] # the index of the track that interact with blocking

# Z500 anomaly data
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
Zanom_origin = np.array(ds['z'].squeeze())  # [0]~[-1] it's from north to south
print(np.shape(Zanom_origin), flush=True)
print('-------- Zanom loaded --------', flush=True)
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
    Zanom = Zanom_origin[:, 0:lat_mid-1, :]
else:
    Zanom = Zanom_origin[:, lat_mid:len(lat), :]

# %% 03 get the target blocking events -------------------
# get the timepoint when the track enter the blocking event for each track events
intopercentList = np.load(f'/scratch/bell/hu1029/LGHW/intopercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy')
targetTrackID = InteractingBlockID_positive[np.where(intopercentList<=0.1)[0]] # the track id that enter the blocking event within the first 20% of the blocking event duration
targetBlockID = InteractingBlockID[targetTrackID]
nontargetTrackID = InteractingBlockID_positive[np.where(intopercentList>0.1)[0]] # the track id that enter the blocking event after the first 20% of the blocking event duration
nontargetBlockID = InteractingBlockID[nontargetTrackID]
print(f'target track ids: {targetTrackID}', flush=True)
print(f'target block ids: {targetBlockID}', flush=True)
havetrackblkIDs_all = np.unique(InteractingBlockID[InteractingBlockID>=0])

blkEventIDs_all = np.unique(blockingEidArr[blockingEidArr>=0])
notrackblkIDs = [bid for bid in blkEventIDs_all if bid not in havetrackblkIDs_all]
print('length of notrack blocks:', len(notrackblkIDs), flush=True)

# # find the first day and -1 day of the blocking event for each notrackblk event
# nontargetindices_notrack = []
# nontargetblkseedDay1_notrack = []
# for jj in range(len(notrackblkIDs)):
#     tgblkid = notrackblkIDs[jj] 
#     print(f'testblkid (notrack): {tgblkid}', flush=True)
#     # find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
#     t_idx = np.where(np.any(blockingEidArr == tgblkid, axis=(1, 2)))[0]
#     firstday = t_idx[0]
#     targetday = firstday - 4
#     nontargetindices_notrack.append(targetday)  # (track id, target day, first day, block id)
#     nontargetblkseedDay1_notrack.append(blockingEidArr[firstday, :, :])  # the first day of the blocking event
# # save results
# with open("/scratch/bell/hu1029/LGHW/nontargetindices_notrack.pkl", "wb") as f:
#     pickle.dump(nontargetindices_notrack, f)
# with open("/scratch/bell/hu1029/LGHW/nontargetblkseedDay1_notrack.pkl", "wb") as f:
#     pickle.dump(nontargetblkseedDay1_notrack, f)

# # find the first day and -1 day of the blocking event for each target track
# targetindices = []
# blkseedDay1 = []
# for jj in range(len(targetBlockID)):
#     tgblkid = targetBlockID[jj] 
#     print(f'testblkid: {tgblkid}', flush=True)
#     # find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
#     t_idx = np.where(np.any(blockingEidArr == tgblkid, axis=(1, 2)))[0]
#     firstday = t_idx[0]
#     targetday = firstday - 4
#     targetindices.append(targetday)  # (track id, target day, first day, block id)
#     blkseedDay1.append(blockingEidArr[firstday, :, :])  # the first day of the blocking event
# # save results
# print('all target indices:', targetindices, flush=True)
# with open("/scratch/bell/hu1029/LGHW/targetindices.pkl", "wb") as f:
#     pickle.dump(targetindices, f)
# with open("/scratch/bell/hu1029/LGHW/blkseedDay1.pkl", "wb") as f:
#     pickle.dump(blkseedDay1, f)

# # same but for non-target events
# nontargetindices = []
# nontargetblkseedDay1 = []
# for jj in range(len(nontargetBlockID)):
#     tgblkid = nontargetBlockID[jj] 
#     print(f'testblkid: {tgblkid}', flush=True)
#     # find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
#     t_idx = np.where(np.any(blockingEidArr == tgblkid, axis=(1, 2)))[0]
#     firstday = t_idx[0]
#     targetday = firstday - 4
#     nontargetindices.append(targetday)  # (track id, target day, first day, block id)
#     nontargetblkseedDay1.append(blockingEidArr[firstday, :, :])  # the first day of the blocking event
# # save results
# with open("/scratch/bell/hu1029/LGHW/nontargetindices.pkl", "wb") as f:
#     pickle.dump(nontargetindices, f)
# with open("/scratch/bell/hu1029/LGHW/nontargetblkseedDay1.pkl", "wb") as f:
#     pickle.dump(nontargetblkseedDay1, f)

# To load the list back from the pickle file
with open("/scratch/bell/hu1029/LGHW/targetindices.pkl", "rb") as f:
    targetindices = pickle.load(f)
with open("/scratch/bell/hu1029/LGHW/blkseedDay1.pkl", "rb") as f:
    blkseedDay1 = pickle.load(f)    
with open("/scratch/bell/hu1029/LGHW/nontargetindices.pkl", "rb") as f:
    nontargetindices = pickle.load(f)
with open("/scratch/bell/hu1029/LGHW/nontargetblkseedDay1.pkl", "rb") as f:
    nontargetblkseedDay1 = pickle.load(f)
with open("/scratch/bell/hu1029/LGHW/nontargetindices_notrack.pkl", "rb") as f:
    nontargetindices_notrack = pickle.load(f)
with open("/scratch/bell/hu1029/LGHW/nontargetblkseedDay1_notrack.pkl", "rb") as f:
    nontargetblkseedDay1_notrack = pickle.load(f)
    
# %% 04 make the composite of seeds ------------------------
latarr = np.arange((0-40),(0+40))
lonarr = np.arange((0-80),(0+40))
seedCompArr = np.zeros((6, len(targetindices), len(latarr), len(lonarr)))  # the composite array for the seeds
for k, tgday in enumerate(targetindices):
    print(f'BlockSeed day index: {tgday}', flush=True)
    # get the day slice of the block seed
    mask = blkseedDay1[k]
    mask0 = (mask>0)  # convert to binary mask
    center0 = np.unravel_index(np.argmax(LWA_td[tgday+4,:,:] * mask0, axis=None), LWA_td[tgday,:,:].shape)
    centerlat = center0[0]  # the latitude index of the center
    centerlon = center0[1]  # the longitude index of the center
    print(f'BlockSeed center: {centerlat}, {centerlon}', flush=True)
    # get the Z500 slices -----------
    for j in [-16,-12,-8,-4,0,4]:  # get -5 to 0 days
        slice_k = getSliceSingle(tgday+j, centerlat, centerlon, latLWA, lonLWA, Zanom)  # get the Z500 slice
        seedCompArr[j//4+4,k,:,:] = slice_k
    
centeredEmbryo = np.nanmean(seedCompArr, axis=1)  # average over the event dimension

# make the plot for each day
fig, axes = plt.subplots(6, 1, figsize=(12, 48))  
for i in range(6):
    ax = axes[i]
    cs = ax.contour(lonarr, latarr, centeredEmbryo[i,:,:],
                    levels=np.arange(-60,240,30), colors='k', linewidths=4)
    ax.clabel(cs, inline=True, fontsize=20, fmt='%1.1f')
    ax.set_title(f"Day {i-5}")
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  
plt.savefig(f'BlockSeed_centerComposites_{rgname}_5daysBefore.png', dpi=300)
plt.show()
plt.close()


# %% 05 plot for the non-target events
latarr = np.arange((0-40),(0+40))
lonarr = np.arange((0-80),(0+40))
seedCompArr_nontarget = np.zeros((6, len(nontargetindices), len(latarr), len(lonarr)))  # the composite array for the seeds
for k, tgday in enumerate(nontargetindices):
    print(f'Non-track BlockSeed day index: {tgday}', flush=True)
    # get the day slice of the block seed
    mask = nontargetblkseedDay1[k]
    mask0 = (mask>0)  # convert to binary mask
    center0 = np.unravel_index(np.argmax(LWA_td[tgday+4,:,:] * mask0, axis=None), LWA_td[tgday,:,:].shape)
    centerlat = center0[0]  # the latitude index of the center
    centerlon = center0[1]  # the longitude index of the center
    print(f'Non-track BlockSeed center: {centerlat}, {centerlon}', flush=True)
    # get the Z500 slices -----------
    for j in [-16,-12,-8,-4,0,4]:  # get -5 to 0 days
        slice_k = getSliceSingle(tgday+j, centerlat, centerlon, latLWA, lonLWA, Zanom)  # get the Z500 slice
        seedCompArr_nontarget[j//4+4,k,:,:] = slice_k  

centeredEmbryo_nontarget = np.nanmean(seedCompArr_nontarget, axis=1)  # average over the event dimension

# make the plot for each day
fig, axes = plt.subplots(6, 1, figsize=(12, 48))  
for i in range(6):
    ax = axes[i]
    cs = ax.contour(lonarr, latarr, centeredEmbryo_nontarget[i,:,:],
                    levels=np.arange(-60,240,30), colors='k', linewidths=4)
    ax.clabel(cs, inline=True, fontsize=20, fmt='%1.1f')
    ax.set_title(f"Day {i-5}")
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  
plt.savefig(f'BlockSeed_centerComposites_nontarget_{rgname}_5daysBefore.png', dpi=300)
plt.show()
plt.close()


# %% 06 plot for the non-track events
latarr = np.arange((0-40),(0+40))
lonarr = np.arange((0-80),(0+40))
seedCompArr_nontarget_notrack = np.zeros((6, len(nontargetindices_notrack), len(latarr), len(lonarr)))  # the composite array for the seeds
for k, tgday in enumerate(nontargetindices_notrack):
    print(f'Non-track BlockSeed day index: {tgday}', flush=True)
    # get the day slice of the block seed
    mask = nontargetblkseedDay1_notrack[k]
    mask0 = (mask>0)  # convert to binary mask
    center0 = np.unravel_index(np.argmax(LWA_td[tgday+4,:,:] * mask0, axis=None), LWA_td[tgday,:,:].shape)
    centerlat = center0[0]  # the latitude index of the center
    centerlon = center0[1]  # the longitude index of the center
    print(f'Non-track BlockSeed center: {centerlat}, {centerlon}', flush=True)
    # get the Z500 slices -----------
    for j in [-16,-12,-8,-4,0,4]:  # get -5 to 0 days
        slice_k = getSliceSingle(tgday+j, centerlat, centerlon, latLWA, lonLWA, Zanom)  # get the Z500 slice
        seedCompArr_nontarget_notrack[j//4+4,k,:,:] = slice_k

centeredEmbryo_nontarget_notrack = np.nanmean(seedCompArr_nontarget_notrack, axis=1)  # average over the event dimension

# make the plot for each day
fig, axes = plt.subplots(6, 1, figsize=(12, 48))  
for i in range(6):
    ax = axes[i]
    cs = ax.contour(lonarr, latarr, centeredEmbryo_nontarget_notrack[i,:,:],
                    levels=np.arange(-60,240,30), colors='k', linewidths=4)
    ax.clabel(cs, inline=True, fontsize=20, fmt='%1.1f')
    ax.set_title(f"Day {i-5}")
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  
plt.savefig(f'BlockSeed_centerComposites_nontarget_notrack_{rgname}_5daysBefore.png', dpi=300)
plt.show()
plt.close() 


