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
import dask

# %% 00 prepare the environment and functions
regions = ["ATL", "NP", "SP"]
seasons = ["ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]]
blkTypes = ["Ridge", "Trough", "Dipole"]

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

# %% 01 prepare the data --------------------------------------------------------------
# get the Z500anom, 1dg (same as LWA)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc')
Zanom_origin = np.array(ds['z'].squeeze())  # lat increasing
print(np.shape(Zanom_origin), flush=True)
print('-------- Zanom loaded --------', flush=True)
# attributes for tracks
lon = np.array(ds['lon'])
lat = np.array(ds['lat']) # -90~90, lat increasing
print(lat, flush=True)

# time management -----------
times = np.array(ds['time'])
datetime_array = pd.to_datetime(times)
timei = list(datetime_array)

# %% get the composites centered at the peak
for ss in seasons:
    for typeid in [1,2,3]:
        for rgname in regions:
        
            # 01 read the data and attributes
            # attributes for track array and blockings
            lat_mid = int(len(lat)/2) + 1 
            if rgname == "SP":
                Blklat = lat[0:lat_mid-1]
                Zanom = Zanom_origin[:, 0:lat_mid-1, :]
            else:
                Blklat = lat[lat_mid:len(lat)]
                Zanom = Zanom_origin[:, lat_mid:len(lat), :]
            tracklat = Blklat
            print('Blocking and track lats: ',tracklat, flush=True)
            tracklon = lon 
            
            # track points
            # check if the file exists, if not, skip this loop
            if not os.path.exists(f'/scratch/bell/hu1029/LGHW/CCtrackInteracting_array_Type{typeid}_{rgname}_{ss}.npy'):
                print(f'File for Type{typeid}_{rgname}_{ss} does not exist, skipping...', flush=True)
                continue
            if not os.path.exists(f'/scratch/bell/hu1029/LGHW/ACtrackInteracting_array_Type{typeid}_{rgname}_{ss}.npy'):
                print(f'File for Type{typeid}_{rgname}_{ss} does not exist, skipping...', flush=True)
                continue
            CCtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/CCtrackInteracting_array_Type{typeid}_{rgname}_{ss}.npy')
            ACtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/ACtrackInteracting_array_Type{typeid}_{rgname}_{ss}.npy')
            print('ACtrackpoints Values',np.unique(ACtrackpoints), flush=True)
            print('CCtrackpoints Values', np.unique(CCtrackpoints), flush=True)
            
            CCtrackpoints = CCtrackpoints.astype(float)
            ACtrackpoints = ACtrackpoints.astype(float)
            print('all trackpoints loaded', flush=True)

            # get the 1st date and location of blocking events
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                peakdateIndex = pickle.load(fp)
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLatList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                peakdatelatV = pickle.load(fp)
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLonList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                peakdatelonV = pickle.load(fp)
            print('peakblocking date and location loaded', flush=True)
            print(len(peakdateIndex))
            print(len(peakdatelatV))
            print(len(peakdatelonV))

            # use the peaking index to select the center LWA region using relative lats and lons
            ttblklen = len(peakdateIndex)
            centeredZ500 = []
            centeredAC = []
            centeredCC = []

            def getSlice(LWA_dayi, latid, lonid, latLWA, lonLWA, LWA_td, 
                         latup=40, latdown=40, lonleft=80, lonright=40, timeextend=32):
                
                LWAlatStart = latid-latdown if latid-latdown >= 0 else 0 
                LWAlatEnd = latid+latup if latid+latup <= len(latLWA) else len(latLWA)
                
                # make sure the same target length
                time_dim = LWA_td.shape[0]
                start_idx = LWA_dayi - timeextend
                end_idx   = LWA_dayi + timeextend + 1 
                valid_start = max(start_idx, 0)
                valid_end   = min(end_idx, time_dim)
                pad_before = valid_start - start_idx if start_idx < 0 else 0
                pad_after  = end_idx - valid_end if end_idx > time_dim else 0
                valid_slice = LWA_td[valid_start:valid_end, LWAlatStart:LWAlatEnd, :]
                LWALatSlice = np.pad(valid_slice, 
                                        pad_width=((pad_before, pad_after), (0, 0), (0, 0)), 
                                        mode='constant', constant_values=np.nan)
                
                if latid-latdown < 0:
                    num_new_rows = int(abs(latid-latdown))
                    new_rows = np.full((LWALatSlice.shape[0], num_new_rows, LWALatSlice.shape[2]), np.nan)  
                    LWALatSlice = np.concatenate((new_rows,LWALatSlice),axis=1)
                    print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
                if latid+latup > len(latLWA):
                    num_new_rows = int(abs(latid+latup-len(latLWA)))
                    new_rows = np.full((LWALatSlice.shape[0], num_new_rows, LWALatSlice.shape[2]), np.nan)
                    LWALatSlice = np.concatenate((LWALatSlice,new_rows),axis=1)
                    print(f'add new rows to the top, now LWAslice shape: {LWALatSlice.shape}',flush=True)
                # lon
                start = lonid - lonleft
                end = lonid + lonright
                if start < 0:  
                    indices = list(range(start + len(lonLWA), len(lonLWA))) + list(range(0, end))
                elif end >= len(lonLWA):
                    indices = list(range(start, len(lonLWA))) + list(range(0, end - len(lonLWA)))
                else:
                    indices = list(range(start, end))
                slice_LWA = LWALatSlice[:,:,indices]

                return slice_LWA

            for i in range(ttblklen):

                track_dayi = timei.index(peakdateIndex[i])
                # get the center lat and lon location (index)
                latTrackid = findClosest(peakdatelatV[i], tracklat)  # get the index of the lat
                lonTrackid = findClosest(peakdatelonV[i], tracklon)  # get the index of the lon

                print(peakdatelatV[i], peakdatelonV[i], 'latid:', latTrackid, 'lonid:', lonTrackid,)

                # get the LWA slices -----------
                slice_Z500 = getSlice(track_dayi, latTrackid, lonTrackid, tracklat, tracklon, Zanom)  # get the Z500 slice
                slice_AC = getSlice(track_dayi, latTrackid, lonTrackid, tracklat, tracklon, ACtrackpoints)  # get the trackpoints
                slice_CC = getSlice(track_dayi, latTrackid, lonTrackid, tracklat, tracklon, CCtrackpoints)  # get the trackpoints

                centeredZ500.append(slice_Z500)
                centeredAC.append(slice_AC)  
                centeredCC.append(slice_CC)  

            centeredZ500npy = np.array(centeredZ500, dtype=np.float32)
            centeredACnpy = np.array(centeredAC, dtype=np.float32)
            centeredCCnpy = np.array(centeredCC, dtype=np.float32)
            
            # four dimensions: event, relativetime(len=41), relativelat, relativelon

            # latarr = np.arange((0-40),(0+40))
            # lonarr = np.arange((0-80),(0+40))
            # _, AC_lat_idx, AC_lon_idx = np.where(centeredACnpy[:,20,:,:] >= 1)
            # _, CC_lat_idx, CC_lon_idx = np.where(centeredCCnpy[:,20,:,:] >= 1)
            # centeredLWAnpyti = np.nanmean(centeredZ500npy, axis=0)[20,:,:]
            # fig, ax = plt.subplots(figsize=(8, 6))
            # cs = ax.contour(lonarr, latarr, centeredLWAnpyti, levels=10, colors='k', linewidths=1.5)
            # ax.scatter(lonarr[AC_lon_idx], latarr[AC_lat_idx], c='blue', marker='o', s=90, edgecolors='none', alpha=0.6)
            # ax.scatter(lonarr[CC_lon_idx], latarr[CC_lat_idx], c='red', marker='o', s=90, edgecolors='none', alpha=0.7)
                
            # ax.clabel(cs, inline=True, fontsize=12)
            # # plt.scatter(0, 0, color='red', s=100)
            # ax.set_xlabel('relative longitude')
            # ax.set_ylabel('relative latitude')
            # plt.show()
            # plt.savefig(f'testcentermap_{typeid}_{rgname}_{ss}.png', dpi=300)
            # plt.close()

            # print('test fig saved', flush=True)

            np.save(f'/scratch/bell/hu1029/LGHW/CenteredZ500_timewindow65_BlkType_Type{typeid}_{rgname}_{ss}.npy', centeredZ500npy)
            np.save(f'/scratch/bell/hu1029/LGHW/CenteredAC_timewindow65_BlkType_Type{typeid}_{rgname}_{ss}.npy', centeredACnpy)
            np.save(f'/scratch/bell/hu1029/LGHW/CenteredCC_timewindow65_BlkType_Type{typeid}_{rgname}_{ss}.npy', centeredCCnpy)

            print('centered composites saved for typeid:', typeid, 'region:', rgname, 'season:', ss, flush=True)

print('composites done')

# Figure - all composites
for ss in seasons:
    for rgname in regions:

        timestep = np.arange(-32,33,8)
        print(timestep, flush=True)

        ncols, nrows = 3, len(timestep)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3))

        latarr = np.arange((0-40),(0+40))
        lonarr = np.arange((0-80),(0+40))
        titles = ['Ridge', 'Trough', 'Dipole']

        # axes is a 2D array with shape (11, 3)
        for i,tid in enumerate(timestep):
            ti = tid + 32  # Adjust index to match the actual position in the array (tid = -32 to 32 maps to ti = 0 to 64)
            for j in range(ncols):

                typeid = j+1 # typeid: 1,2,3

                centeredZ500npy = np.load(f'/scratch/bell/hu1029/LGHW/CenteredZ500_timewindow65_BlkType_Type{typeid}_{rgname}_{ss}.npy')
                centeredACnpy = np.load(f'/scratch/bell/hu1029/LGHW/CenteredAC_timewindow65_BlkType_Type{typeid}_{rgname}_{ss}.npy')
                centeredCCnpy = np.load(f'/scratch/bell/hu1029/LGHW/CenteredCC_timewindow65_BlkType_Type{typeid}_{rgname}_{ss}.npy')

                centeredZ500npyti = np.nanmean(centeredZ500npy, axis=0)[ti,:,:]  # average over the time dimension
                _, AC_lat_idx, AC_lon_idx = np.where(centeredACnpy[:,ti,:,:] >= 1)
                _, CC_lat_idx, CC_lon_idx = np.where(centeredCCnpy[:,ti,:,:] >= 1)
                
                ax = axes[i, j]
                cs = ax.contour(lonarr, latarr, centeredZ500npyti, levels=10, colors='k', linewidths=1.5)
                ax.scatter(lonarr[AC_lon_idx], latarr[AC_lat_idx], c='#0044FF', marker='o', s=90, edgecolors='none', alpha=0.4)
                ax.scatter(lonarr[CC_lon_idx], latarr[CC_lat_idx], c='#FF4800', marker='o', s=90, edgecolors='none', alpha=0.4)
                ax.clabel(cs, inline=True, fontsize=10)

                # row title
                if j == 0:
                    ax.set_ylabel(f'Day {tid/4:.0f}', fontsize=12)
                # column title
                if i == 0:
                    ax.set_title(titles[j], fontsize=13)

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        plt.savefig(f'centerComposites_allpanels_1stBlkDay_extendedLon_3Types_{rgname}_{ss}.png', dpi=300)
        plt.show()
        plt.close()

        print('done with region:', rgname, 'season:', ss, flush=True)
