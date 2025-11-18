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
seasons = [ "ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["AC","CC"]

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

def extendList(desired_length,centorinlist,test):
    targetlist = np.full(desired_length*2+1, np.nan)
    targetlist[desired_length] = test[centorinlist]
    leftlen = centorinlist
    rightlen = len(test) - leftlen -1
    left_padding = [np.nan] * (desired_length - leftlen)
    left_values = left_padding + list(test[np.nanmax([0, leftlen-desired_length]):centorinlist])
    right_padding = [np.nan] * (desired_length - rightlen)
    if rightlen-desired_length>0:
        rightend = centorinlist+1+desired_length
    else: rightend = len(test)
    right_values = list(test[centorinlist+1:rightend]) + right_padding
    targetlist[:desired_length] = left_values
    targetlist[desired_length+1:] = right_values
    return targetlist

# %% 01 get the data ------------------------

# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in lat and lon for LWA
lon = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
lat = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order 90~-90

ss = 'ALL' 
for typeid in [1,2,3]: 
    for cyc in cycTypes:
        for rgname in regions:            

                # get the lat and lon for LWA, grouped by NH and SH
                lat_mid = int(len(lat)/2) + 1 
                if rgname == "SP":
                    latLWA = lat[lat_mid:len(lat)]
                else:
                    latLWA = lat[0:lat_mid-1]
                latLWA = np.flip(latLWA) # make it ascending order (from south to north)
                print(latLWA, flush=True)
                lonLWA = lon 

                blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/SD_BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
                # transfer to 6-hourly
                blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
                blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

                # read in the event's LWA list
                with open(f"/scratch/bell/hu1029/LGHW/BlockEventDailyLWAList_1979_2021_Type{typeid}_{rgname}_{ss}.pkl", "rb") as f:
                    BlkeventLWA = pickle.load(f) 

                # the id of each blocking event (not all events! just the events within the target region)
                with open(f'/scratch/bell/hu1029/LGHW/SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
                    blkEindex = pickle.load(f)
                blkEindex = np.array(blkEindex)

                # read in the track's interaction information
                if rgname == "SP":
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021_SH.pkl', 'rb') as file:
                        LWAtrack = pickle.load(file)
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_SH.pkl', 'rb') as file:
                        track_data = pickle.load(file)
                else:
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021_NH.pkl', 'rb') as file:
                        LWAtrack = pickle.load(file)
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_NH.pkl', 'rb') as file:
                        track_data = pickle.load(file)

                # EddyNumber = np.load(f'/scratch/bell/hu1029/LGHW/BlockingType{typeid}_EventEddyNumber_1979_2021_{rgname}_{ss}_{cyc}.npy')
                eddyBlockIndex = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')

                # get the enter/leaving time for each eddy [the location relative to the blocking]
                entertime = np.load(f'/scratch/bell/hu1029/LGHW/EnterTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
                leavetime = np.load(f'/scratch/bell/hu1029/LGHW/LeaveTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
                                
                # %% 02 test plotting for one event ------------------------
                interactingBlkID = eddyBlockIndex
                interactingBlkID = interactingBlkID[np.where(interactingBlkID >= 0)]
                unique_ids, counts = np.unique(interactingBlkID, return_counts=True) # counts is the number of eddies in each block (unique_ids)

                lenbf = 8; lenaf = 8
                lastlimit = 8 # at least 2 days left
                beginlimit = 8 # at least 2 days after the blocking start
                TrackAVGLWAlist = []
                BlkAVGLWAlist = []
                N = 0
                for ki in interactingBlkID: # 
                    
                    print(f'Processing blocking id: {ki}', flush=True)
                    testblkid = ki #

                    # the time Dates for the target blocking event
                    maskday = np.any(blockingEidArr == testblkid, axis=(1, 2))  #shape=(time,) 
                    blkglobalLoc = np.where(maskday)[0]
                    blklastday = blkglobalLoc[-1] # the last day of the blocking event
                    blk1stday = blkglobalLoc[0] # the first day of the blocking event
                    BlkLWA = BlkeventLWA[np.where(blkEindex==testblkid)[0][0]] # get the LWA series for the target blocking event

                    # find the eddy indices for the target blocking event
                    eddyindices = np.where(eddyBlockIndex == testblkid)[0] # the eddy indices for the target blocking event
                    targetTracks = [track_data[i] for i in eddyindices] # the track data for the target blocking event
                    TrackLWAlist = [LWAtrack[i] for i in eddyindices] # the LWA list for the target blocking event
                    TrackDates = [np.array([ti for ti, _, _ in pointlist]) for _, pointlist in targetTracks]
                    # print('track dates:', TrackDates, flush=True)
                    # find the enter relative timeindex
                    for i in range(len(TrackDates)):
                        entertimei = entertime[eddyindices[i]] # the relative index
                        # the real enter time index 
                        realtimeenter = TrackDates[i][entertimei] # the date of the eddy entering the blocking
                        entertimeglobalLoc = timei.index(realtimeenter) # the global index of the eddy entering the blocking
                        if blklastday - entertimeglobalLoc < lastlimit or entertimeglobalLoc - blk1stday < beginlimit:
                            continue # if the eddy entering too early/late (less than 8 timesteps left), skip it
                        
                        # re-organize the LWA
                        blkcenter = np.where(blkglobalLoc == entertimeglobalLoc)[0][0]  # the center of the blocking event
                        # print(f'Blocking center index: {blkcenter}', flush=True)
                        # print(f'Eddy {i} enter time index: {entertimei}', flush=True)
                        track_lwa = TrackLWAlist[i]  # the LWA series for the eddy track
                        tracklwarecenter = extendList(lenaf, entertimei, track_lwa) # re-organize the LWA data
                        blklwarecenter = extendList(lenaf, blkcenter, BlkLWA) # re-organize the LWA data
                        # print('eddy original:', track_lwa, flush=True)
                        # print('eddy recenter: ', tracklwarecenter, flush=True)
                        # print('blocking original:', BlkLWA, flush=True)
                        # print('blocking recenter: ',blklwarecenter, flush=True)

                        TrackAVGLWAlist.append(tracklwarecenter) # calculate the average LWA for the track
                        BlkAVGLWAlist.append(blklwarecenter) # calculate the average LWA for the blocking

                        N += 1 # add one interaction event

                # save the AVG LWA series
                np.save(f'/scratch/bell/hu1029/LGHW/MiddleEddiesTrackLWAseries_Type{typeid}_{rgname}_{ss}_{cyc}_composites.npy', np.array(TrackAVGLWAlist))
                np.save(f'/scratch/bell/hu1029/LGHW/MiddleEddiesBlkLWAseries_Type{typeid}_{rgname}_{ss}_{cyc}_composites.npy', np.array(BlkAVGLWAlist))

                TrackAVGLWAarr = np.nanmean(np.array(TrackAVGLWAlist), axis=0) # average LWA for all tracks
                BlkAVGLWAarr = np.nanmean(np.array(BlkAVGLWAlist), axis=0) # average LWA for all blockings

                # plot: the process line of the LWA
                relativetimestep = np.arange(-lenaf, lenaf+1)  # relative time step from -19 to 19
                fig, ax1 = plt.subplots(figsize=(10, 6))
                # Plot blocking LWA on left y-axis
                ax1.fill_between(relativetimestep, BlkAVGLWAarr, color='tab:blue', alpha=0.3, label='Blocking LWA')
                ax1.plot(relativetimestep, BlkAVGLWAarr, color='tab:blue', linewidth=2)  
                ax1.set_xlabel('Relative Time Step (6-hourly)', fontsize=14)
                ax1.set_ylabel('Blocking LWA', color='tab:blue', fontsize=14)
                ax1.tick_params(axis='x', labelsize=12)
                ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
                ax1.text(0.02, 0.98, f'Interaction Number = {N}', transform=ax1.transAxes,
                            ha='left', va='top', fontsize=16)
                # Create right y-axis for track LWA
                cols = ['orange', 'orangered', 'red']  # colors for different tracks
                ax2 = ax1.twinx()
                ax2.plot(relativetimestep, TrackAVGLWAarr, label=f'Track LWA',color='orangered', marker='o')
                ax2.axvline(0, color='orangered', linestyle='--', linewidth=1.5, label='Enter Time')
                ax2.set_ylabel('Track LWA', color='orangered', fontsize=14)
                ax2.tick_params(axis='y', labelcolor='orangered', labelsize=12)

                # # Legends
                # lines1, labels1 = ax1.get_legend_handles_labels()
                # lines2, labels2 = ax2.get_legend_handles_labels()
                # ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                plt.tight_layout()
                plt.show()
                plt.savefig(f'./MiddleEddiesLWAseries_Type{typeid}_{rgname}_{ss}_{cyc}_composites.png', dpi=300, bbox_inches='tight')

                print(f'Fig saved: Type{typeid}_{rgname}_{ss}_{cyc}', flush=True)
