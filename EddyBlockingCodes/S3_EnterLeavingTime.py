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
from scipy.stats import pearsonr

# %% 00 prepare the environment and functions
regions = ["ATL", "NP", "SP"]
seasons = ["ALL", "DJF", "JJA"]
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

for rgname in regions:
    
    # 01 - read data -------------------------------------------------------------
    # lon and lat for the track
    ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
    lon = np.array(ds['lon'])
    lat = np.array(ds['lat'])
    lat = np.flip(lat)
    if rgname == "SP":
        lat = lat[0:(findClosest(0,lat)+1)] # lat increasing!
    else:
        lat = lat[(findClosest(0,lat)+1):len(lat)]
    # time management
    times = np.array(ds['time'])
    datetime_array = pd.to_datetime(times)
    timei = list(datetime_array)
    # lat and lon for blocking
    lonBLK = np.load('/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy')
    latBLK = np.load('/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy') # descending order
    lat_mid = int(len(latBLK)/2) + 1 
    if rgname == "SP":
        latBLK = latBLK[lat_mid:len(latBLK)]
    else:
        latBLK = latBLK[0:lat_mid-1]
    latBLK = np.flip(latBLK) # make the lat increasing
    print(latBLK)

    for typeid in [1,2,3]:
        for cyc in cycTypes:
            for ssi,ss in enumerate(seasons):

                target_months = seasonsmonths[ssi] # the target months for each season

                print(f'start: Blocking type{typeid} - {cyc} - {rgname} - {ss}')

                # get all the tracks
                # load tracks
                if rgname == "SP":
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_SH.pkl', 'rb') as file:
                        track_data = pickle.load(file)
                else:
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_NH.pkl', 'rb') as file:
                        track_data = pickle.load(file)
                print('track loaded-----------------------',flush=True)

                lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname) # get the region range

                # get the blocking index each track contribute to (-1 or the blocking global index), 1d, same length as the track_data
                InteractingBlockID = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')        
                # get the blocking event index list (global index), 1d list, all blocking events in the target region
                with open(f'/scratch/bell/hu1029/LGHW/SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
                    Sec2BlockEvent = pickle.load(f)
                print(f'blocking id of each event: {Sec2BlockEvent}',flush=True)
                eventPersistence = np.load(f'/scratch/bell/hu1029/LGHW/BlockingEventPersistence_Type{typeid}_{rgname}_{ss}.npy') # load the event persistence
                # the 1st day's date
                with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
                    firstday_Date = pickle.load(fp)

                # the arr of the blocking event, filled witht the event id, 3d
                BlockingEIndexSec2 = np.load(f'/scratch/bell/hu1029/LGHW/SD_BlockingFlagmaskClusters_Type{typeid}_{rgname}_{ss}.npy')
                BlockingEIndexSec2 = np.flip(BlockingEIndexSec2, axis=1) # flip the lat, make it increasing
                BlockingEIndexSec2 = np.repeat(BlockingEIndexSec2, 4, axis=0)

                # 01 the number of blocking events within the target region --------------------------------
                numBlockEvent = len(Sec2BlockEvent)
                BlockwithTrack = 0
                BlockwithoutTrack = 0
                InteractingBlockID_uniquev = InteractingBlockID[InteractingBlockID != -1] # remove the -1
                InteractingBlockID_uniquev = np.unique(InteractingBlockID_uniquev) # get the unique values
                BlockwithTrackLen = len(InteractingBlockID_uniquev) # the number of unique blocking events that has ever interact with the tracks

                with open(f"./SD_enterleave/BlockingBasedProbability.txt", "a") as f:  
                    f.write(f'Blocking type{typeid} - {cyc} - {rgname} - {ss} withTrack number: {BlockwithTrackLen}\n')
                    f.write(f'Blocking type{typeid} - {cyc} - {rgname} - {ss} withoutTrack number: {numBlockEvent-BlockwithTrackLen}\n')

                # 02 the number of all the tracks that has ever enter into the target region --------------------------------
                lat_indices = (lat >= lat_min) & (lat <= lat_max)
                lon_indices = (lon >= lon_min) & (lon <= lon_max)
                if lon_min > lon_max:
                    lon_indices = (lon >= lon_min) | (lon <= lon_max)
                lat_indices = np.where(lat_indices)[0]
                lon_indices = np.where(lon_indices)[0]
                lat_indices, lon_indices = np.ix_(lat_indices, lon_indices)
                region_mask = np.zeros((len(lat), len(lon)), dtype=bool)
                region_mask[lat_indices, lon_indices] = True

                n = 0
                for _, pointlist in track_data:
                    latids = [lati for _, _, lati in pointlist]
                    latids = findClosest(latids, lat)
                    lonids = [loni for _, loni, _ in pointlist]
                    lonids = findClosest(lonids, lon)
                    trackdates = [ti for ti, _, _ in pointlist] # the time index in the real all time list
                    if all(date.month not in target_months for date in trackdates): # check if within the target seasons
                        continue
                    for k in range(len(latids)):
                        if region_mask[latids[k], lonids[k]]:
                            n = n+1
                            break # count only once for each track
                
                with open(f"./SD_enterleave/EddyBasedProbability.txt", "a") as f:  
                    f.write(f'Eddy type{typeid} - {cyc} - {rgname} - {ss} has ever enter into the target region: {n}\n')
                    f.write(f'Eddy type{typeid} - {cyc} - {rgname} - {ss} has ever interact with blocking: {np.sum(InteractingBlockID != -1)}\n')
                    f.write('------------------------\n')

                # 03 calculate the distribution of entering and leaving time of the blocking events, relative to the blocking's life cycle --------------------------------
                intopercentList = []
                leavepercentList = []
                blkpersistList = []
                entertime = []
                leavetime = []
                interactingduration = []
                # i is the track's oreder, blockid is the relevant blocking event id
                for i,blockid in enumerate(InteractingBlockID):
                    # blockid is the global blocking event id, -1 means no blocking event
                    et = -1; lt = -1 # the first into time and last leave time
                    if blockid >=0:
                        
                        blockinglocation = Sec2BlockEvent.index(blockid) # the location of the blocking event in the Sec2BlockEvent list
                        # the event's persistence
                        evps = eventPersistence[blockinglocation] # the persistence of the blocking event
                        blkpersistList.append(evps)

                        # get the track's information
                        track = track_data[i] # the target track's information
                        # tracklwa = LWAtrack_Sec2[i] # the target track's LWA
                        _, pointlist = track # get the point list
                        times = [ti for ti, _, _ in pointlist]
                        latids = [lati for _, _, lati in pointlist]
                        latids = findClosest(latids, latBLK)
                        lonids = [loni for _, loni, _ in pointlist]
                        lonids = findClosest(lonids, lonBLK)
                        timeids = [timei.index(j) for j in times] # the time index in the real all time list

                        # get the blocking event's information
                        blockingValue = BlockingEIndexSec2[np.array(timeids), np.array(latids), np.array(lonids)] 
                        # True or False
                        print(blockingValue) # [0,0,...,1,1,1,0,0,...]
                        firstinto = np.argmax(blockingValue) # the first index of blockingValue == True
                        lastleave = np.where(blockingValue)[0][-1] # the last index of blockingValue == True
                        intertime = np.nansum(blockingValue) # the total number of interacting time steps
                        # find these two points in the realtimeindex (relative position)
                        firstinto_realtime = timeids[firstinto] # the location of the first into time 
                        lastleave_realtime = timeids[lastleave] # the location of the last leave time
                        firstday_Date_index = timei.index(firstday_Date[blockinglocation]) # the location of the first day of the blocking event
                        durationtime = evps*4 # the duration time of the blocking event in 6-hourly time steps
                        print(f'firstinto: {firstinto_realtime}, lastleave: {lastleave_realtime}, firstinto_realtime: {firstinto_realtime}, lastleave_realtime: {lastleave_realtime}, durationtime: {durationtime}', flush=True)
                        
                        start_relative_day = firstinto_realtime - firstday_Date_index + 1
                        end_relative_day = lastleave_realtime - firstday_Date_index + 1
                        intopercent = start_relative_day / durationtime
                        leavepercent   = end_relative_day / durationtime
                        print(f'start_relative_day: {start_relative_day}, end_relative_day: {end_relative_day}, intopercent: {intopercent}, leavepercent: {leavepercent}', flush=True)

                        intopercentList.append(intopercent)
                        leavepercentList.append(leavepercent)
                        interactingduration.append(intertime/4)

                        et = firstinto
                        lt = lastleave

                    entertime.append(et)
                    leavetime.append(lt)

                np.save(f'/scratch/bell/hu1029/LGHW/EnterTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy', entertime)
                np.save(f'/scratch/bell/hu1029/LGHW/LeaveTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy', leavetime)

                np.save(f'/scratch/bell/hu1029/LGHW/intopercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy', np.array(intopercentList))
                np.save(f'/scratch/bell/hu1029/LGHW/leavepercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy', np.array(leavepercentList))
                np.save(f'/scratch/bell/hu1029/LGHW/interactingduration_type{typeid}_{cyc}_{rgname}_{ss}.npy', np.array(interactingduration))

                print('Enter and leave time saved-----------------------',flush=True)

                # # plot 01: KDE+scatter plot of entry time and blocking duration+blocking duration pdf
                # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                # fig, ax1 = plt.subplots(figsize=(10, 6))

                # sns.kdeplot(
                #     x=interactingduration,  # convert to days
                #     y=intopercentList, 
                #     fill=True, 
                #     cmap='Reds', 
                #     levels=50, 
                #     thresh=0.05,
                #     cbar=True,
                #     ax=ax1,
                #     cbar_kws={'label': 'Density'}
                # )
                # ax1.set_xlabel('Eddy Stay time (days)')
                # ax1.set_ylabel('Entry Time (% of Blocking Persistence)')
                # ax1.set_ylim(0, 1)
                # ax1.set_title(f'Eddy-Blocking Interaction Number: {len(intopercentList)}')
                # plt.savefig(f'./SD_enterleave/EnterWithStay_type{typeid}_{cyc}_{rgname}_{ss}.png')
                # plt.close()

                # plot 02: make a 1d histogram of the intopercentList
                fig, ax1 = plt.subplots(figsize=(8, 6))
                # Plot histogram as frequency (count)
                sns.histplot(intopercentList, bins=20, stat='count', ax=ax1, color='skyblue')
                ax1.set_xlabel('Entry Time (% of Blocking Persistence)')
                ax1.set_ylabel('Frequency')
                ax1.set_title(f'Eddy-Blocking Interaction Number: {len(intopercentList)}')
                ax1.set_xlim(0, 1)
                plt.savefig(f'./SD_enterleave/EnterTimePDF_type{typeid}_{cyc}_{rgname}_{ss}.png')
                plt.close()


# statistics of the entering time and interacting duration
ss = 'ALL'
for typeid in [1,2,3]:
     for cyc in cycTypes:
        for rgname in regions:

            print(f'start: Blocking type{typeid} - {cyc} - {rgname} - {ss}')
            intopercentList = np.load(f'/scratch/bell/hu1029/LGHW/intopercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy')
            interactingduration = np.load(f'/scratch/bell/hu1029/LGHW/interactingduration_type{typeid}_{cyc}_{rgname}_{ss}.npy')
            print(f'{typeid}_{cyc}_{rgname}_{ss}, intopercent<=20% percentage: {len(np.where(intopercentList<=0.2)[0])/len(intopercentList)}', flush=True)
            print(f'{typeid}_{cyc}_{rgname}_{ss}, duration<=2 percentage: {len(np.where(interactingduration<=2)[0])/len(interactingduration)}', flush=True)

            with open("EnterStay_Statistic.txt", "a") as f:
                f.write(f'{typeid}_{cyc}_{rgname}_{ss}, intopercent<=20% percentage: {len(np.where(intopercentList<=0.2)[0])/len(intopercentList)}\n')
                f.write(f'{typeid}_{cyc}_{rgname}_{ss}, duration<=2 percentage: {len(np.where(interactingduration<=2)[0])/len(interactingduration)}\n')
                f.write('------------------------\n')

                