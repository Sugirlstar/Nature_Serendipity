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


# %% 00 prepare the environment and functions
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA","ALL"]
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

def PlotBoundary(regionname): 
    if regionname == "ATL": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 250, 90, 350
    elif regionname == "NP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = 30, 90, 80, 280, 180
    elif regionname == "SP": 
        lat_min, lat_max, lon_min, lon_max, loncenter = -90, -30, 130, 330, 230

    return lat_min, lat_max, lon_min, lon_max, loncenter

typeid = 1
rgname = "ATL"
ss = "ALL"
cyc = "AC"

for rgname in ["ATL"]:
    
    # 01 - read data -------------------------------------------------------------
    # lon and lat for the track
    ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
    lon = np.array(ds['lon'])
    lat = np.array(ds['lat'])
    lat = np.flip(lat)
    if rgname == "SP":
        lat = lat[0:(findClosest(0,lat)+1)]
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
    latBLK = np.flip(latBLK)
    print(latBLK)

    for typeid in [1]:
        for cyc in ["AC"]:
            for ssi,ss in enumerate(['ALL']):

                # read in the track's timesteps (6-hourly)
                ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
                timesarr = np.array(ds['time'])
                datetime_array = pd.to_datetime(timesarr)
                timei = list(datetime_array)
                # read in the LWA data
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
                print(latLWA, flush=True)
                print('LWA shape: ', LWA_td.shape, flush=True)
                lonLWA = lon 

                # 6-hourly Blocking event id array
                blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
                # transfer to 6-hourly
                blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
                blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

                print(f'start: Blocking type{typeid} - {cyc} - {rgname} - {ss}')

                # get all the tracks
                # load tracks
                if rgname == "SP":
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_SH.pkl', 'rb') as file:
                        track_data = pickle.load(file)
                else:
                    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks.pkl', 'rb') as file:
                        track_data = pickle.load(file)
                print('track loaded-----------------------',flush=True)

                lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname) # get the region range

                # get the blocking index each track contribute to (-1 or the blocking global index), 1d, same length as the track_data
                InteractingBlockID = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy') 
                # the index of the bloking each track interact with. length =  total length of the tracks
                eddyBlockIndex = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')

                # get the timepoint when the track enter the blocking event for each track events
                entertime = np.load(f'/scratch/bell/hu1029/LGHW/EnterTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy')
                targetTrackID = np.where((entertime <= 6) & (entertime > 0))[0] # the track id that enter the blocking event within the first 2 days
                targetBlockID = InteractingBlockID[targetTrackID]
                print(f'target track ids: {targetTrackID}', flush=True)
                print(f'target block ids: {targetBlockID}', flush=True)

                for jj in range(len(targetBlockID)):

                    testblkid = targetBlockID[jj] # test with the first blocking event id
                    print(f'testblkid: {testblkid}', flush=True)

                    # find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
                    maskday = np.any(blockingEidArr == testblkid, axis=(1, 2))  #shape=(time,) 
                    print('event length:', np.nansum(maskday), flush=True) 
                    true_idx = np.where(maskday)[0]
                    print('true_idx:', true_idx, flush=True) # the index of the blocking event in the blockingEidArr_6hr
                    firstday = true_idx[0] # the first day of the blocking event
                    secondday = true_idx[9] # the second day of the blocking event
                    maskday[firstday-8:firstday] = True # extend the first day to 4 timesteps before
                    maskday[firstday+8:] = False # remove the days after the second day
                    print('new true_idx:', np.where(maskday)[0], flush=True) # the new index of the blocking event in the blockingEidArr_6hr

                    BlkTargetArr = blockingEidArr[maskday]
                    BlkTargetArr = BlkTargetArr>=0 # convert to boolean, for making the contour line
                    # get the target LWA 
                    LWATargetArr = LWA_td[maskday, :, :] # get the LWA for the target date
                    timeBlock = np.array(timei)[np.where(maskday)[0]] # the time Dates for the target blocking event
                    print('target Block event dates:', timeBlock, flush=True)

                    # find the eddy indices for the target blocking event
                    eddyindices = np.where(eddyBlockIndex == testblkid)[0] # the eddy indices for the target blocking event
                    targetTracks = [track_data[i] for i in eddyindices] # the track data for the target blocking event

                    daylen = np.shape(LWATargetArr)[0]
                    TargetperiodTrackList = [] # a list to store the track point tracks
                    for i in range(len(timeBlock)):  
                        theday = timeBlock[i] # the target day
                        track_points = [
                        (index, [(lon, lat) for timeid, lon, lat in points if timeid <= theday ]) 
                        for index, points in targetTracks
                        ]
                        TargetperiodTrackList.append(track_points)
                        print(track_points)

                    print(TargetperiodTrackList[0])
                    print(TargetperiodTrackList[1])
                    print(TargetperiodTrackList[2])         

                    # plot1: the map of the LWA
                    lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
                    lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)
                    maxvalue = np.nanmax(LWATargetArr)
                    cflevels = np.linspace(0, maxvalue, 14) # set the contour levels
                    print('cflevels:', cflevels, flush=True)

                    rows = int(daylen/2)
                    fig, axes = plt.subplots(nrows=rows, ncols=2, 
                                            figsize=(14, rows*3), 
                                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=loncenter)})
                    cf_last = None
                    
                    i = 0
                    for col in range(2):              
                        for row in range(rows):     
                            
                            ax = axes[row, col]

                            indicator = i - 8

                            _, _, cf = create_Map(lonLWA, latLWA, LWATargetArr[i, :, :], fill=True, fig=fig, ax=ax,
                                                leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                                minv=0, maxv=maxvalue, cflevels=cflevels, figsize=(12, 5),
                                                centralLon=loncenter, colr='PuBu', extend='max',
                                                title=f'Blocking LWA on {timeBlock[i]}')
                            cf_last = cf
                            ax.set_title(f'LeadDay: {indicator}', fontsize=10, loc='left')
                            ax.contour(lonLWA, latLWA, BlkTargetArr[i, :, :].astype(float), levels=[0.5], 
                                    colors='darkblue', linewidths=1.5, transform=ccrs.PlateCarree())
                            addSegments(ax, [(lon_min, lat_min), (lon_max, lat_min), (lon_max, lat_max), 
                                            (lon_min, lat_max), (lon_min, lat_min)], 
                                        colr='black', linewidth=1)
                            for track_id, points in TargetperiodTrackList[i]:
                                addSegments(ax, points, colr='orangered', linewidth=4, alpha=0.4)
                                if len(points) > 0:
                                    end_lon, end_lat = points[-1]
                                    ax.plot(end_lon, end_lat, marker='x', color='orangered', markersize=10, 
                                            transform=ccrs.PlateCarree())
                            
                            i += 1

                    cbar = fig.colorbar(cf_last, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
                    cbar.set_label('LWA')

                    # 调整间距
                    plt.tight_layout()

                    # 保存整张 panel
                    plt.savefig(f'./blockingSeed/BlkSeed_Type{typeid}_{cyc}_{rgname}_EventID{testblkid}_panels.png', 
                                bbox_inches='tight', dpi=300)
                    plt.close()

                        

                    # for i in range(np.shape(LWATargetArr)[0]):

                    #     indicator = i - 8
                    #     print('Plotting day:', i, flush=True)
                    #     fig, ax, cf = create_Map(lonLWA,latLWA,LWATargetArr[i,:,:],fill=True,fig=None,
                    #                             leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                    #                                 minv=0, maxv=maxvalue, cflevels=cflevels, figsize=(12,5),
                    #                                 centralLon=loncenter, colr='PuBu', extend='max',title=f'Blocking LWA on {timeBlock[i]}',)
                    #     ax.contour(lonLWA, latLWA, BlkTargetArr[i,:,:].astype(float), levels=[0.5], 
                    #                     colors='darkblue', linewidths=1.5, transform=ccrs.PlateCarree())
                        
                    #     addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
                    #     plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

                    #     for track_id, points in TargetperiodTrackList[i]:
                    #         addSegments(ax,points,colr='orangered',linewidth=2,alpha=0.2)
                    #         if(len(points) > 0):
                    #             end_lon, end_lat = points[-1]
                    #             ax.plot(end_lon, end_lat, marker='x', color='orangered', markersize=6, 
                    #                     transform=ccrs.PlateCarree())

                    #     plt.show()
                    #     plt.savefig(f'./blockingSeed/BlkSeed_Type{typeid}_{cyc}_{rgname}_EventID{testblkid}_LeadTime{indicator}.png', bbox_inches='tight', dpi=300)
                    #     plt.close()

