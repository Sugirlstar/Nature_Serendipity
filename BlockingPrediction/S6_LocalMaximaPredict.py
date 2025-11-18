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
ss = "ALL"
cyc = "AC"

# read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

for rgname in regions:

    lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
    lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)

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

    # get the region mask
    lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
    rgmask = np.zeros((len(latLWA), len(lonLWA)))
    lat_min_ix = np.argmin(np.abs(latLWA - lat_min))
    lat_max_ix = np.argmin(np.abs(latLWA - lat_max))
    lon_min_ix = np.argmin(np.abs(lonLWA - lon_min))
    lon_max_ix = np.argmin(np.abs(lonLWA - lon_max))
    if lon_min < lon_max:
        rgmask[lat_min_ix:lat_max_ix+1, lon_min_ix:lon_max_ix+1] = 1
        print('Region mask lon range:', lon_min_ix,'-', lon_max_ix+1, flush=True)
    else:
        # if the region crosses the 360/0 degree longitude
        rgmask[lat_min_ix:lat_max_ix+1, lon_min_ix:len(lonLWA)] = 1
        rgmask[lat_min_ix:lat_max_ix+1, 0:lon_max_ix+1] = 1
        print('Region mask lon range:', lon_min_ix, '-',len(lonLWA), '; 0-', lon_max_ix+1, flush=True)
    print('Region mask lat range:', lat_min_ix, '-', lat_max_ix+1, flush=True)

    fig, ax, cf = create_Map(lonLWA,latLWA,LWA_td[0,:,:],fill=True,fig=None,
                            leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                minv=0, maxv=np.nanmax(LWA_td[0,:,:]), interv=11, figsize=(12,5),
                                centralLon=loncenter, colr='PuBu', extend='max',title=f'LWA on day 0')
    ax.contour(lonLWA, latLWA, rgmask.astype(float), levels=[0.5],
                        colors='black', linewidths=2, transform=ccrs.PlateCarree())
    plt.show()
    plt.savefig(f'test_regionMask.png', bbox_inches='tight', dpi=300)
    plt.close()


    # %% 02 find the wave event --------------------
    T = LWA_td.shape[0]
    T4 = T // 4
    LWA_Z = LWA_td[:T4*4]
    LWA_Z = LWA_Z.reshape(T4, 4, len(latLWA), len(lonLWA)).mean(axis=1)

    nlon = len(lonLWA)
    nlat = len(latLWA)
    nday = np.shape(LWA_Z)[0] # the number of days
    LWA_max_lon = np.zeros((nlon*nday))
    for t in np.arange(nday):    
        for lo in np.arange(nlon):
            LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
    Thresh = np.median(LWA_max_lon[:])
    print('Threshold:', Thresh)
    ### Wave Event ###
    WEvent = np.zeros((nday,nlat,nlon),dtype='uint8') 
    WEvent[LWA_Z>Thresh] = 1                    # Wave event daily
    ### connected component-labeling algorithm ###
    num_labels = np.zeros(nday)
    labels = np.zeros((nday,nlat,nlon))
    for d in np.arange(nday):
        num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WEvent[d,:,:], connectivity=4)
    ####### connect the label around 180, since they are labeled separately ########
    ####### but actually they should belong to the same label  ########
    labels_new = copy.copy(labels)
    for d in np.arange(nday):
        if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at either -180 or 179.375, then WEvent don't need to do any connection
            continue
                
        column_0 = np.zeros((nlat,3))       ## WEvent assume there are at most three wave events at column 0 (-180) (actuaaly most of the time there is just one)
        column_end = np.zeros((nlat,3))
        label_0 = np.zeros(3)
        label_end = np.zeros(3)
        
        ## Get the wave event at column 0 (0) ##
        start_lat0 = 0
        for i in np.arange(3):
            for la in np.arange(start_lat0, nlat):
                if labels_new[d,la,0]==0:
                    continue
                if labels_new[d,la,0]!=0:
                    label_0[i]=labels_new[d,la,0]
                    column_0[la,i]=labels_new[d,la,0]
                    if labels_new[d,la+1,0]!=0:
                        continue
                    if labels_new[d,la+1,0]==0:
                        start_lat0 = la+1
                        break 

            ## Get the wave event at column -1 (359) ## 
            start_lat1 = 0
            for j in np.arange(3):
                for la in np.arange(start_lat1, nlat):
                    if labels_new[d,la,-1]==0:
                        continue
                    if labels_new[d,la,-1]!=0:
                        label_end[j]=labels_new[d,la,-1]
                        column_end[la,j]=labels_new[d,la,-1]
                        if labels_new[d,la+1,-1]!=0:
                            continue
                        if labels_new[d,la+1,-1]==0:
                            start_lat1 = la+1
                            break                       
                ## Compare the two cloumns at 0 and 359, and connect the label if the two are indeed connected
                if (column_end[:,i]*column_0[:,j]).mean() == 0:
                    continue                
                if (column_end*column_0).mean() != 0:
                    num_labels[d]-=1
                    if label_0[i] < label_end[j]:
                        labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
                        labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
                    if label_0[i] > label_end[j]:
                        labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
                        labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]

    print('labels -> labels_new, num_labels -> num_labels', flush=True)

    # %% 03 filter the labels --------------------
    k = 0
    regionLabels = np.zeros((nday, nlat, nlon), dtype=np.int64) # the region labels
    regionlabelList = []
    regionlabelcenterLat = []
    regionlabelcenterLon = []
    regionlabelmaxima = []
    for day in np.arange(nday):
        labels_today = labels_new[day]
        lwa_today = LWA_Z[day]
        unique_labels = np.unique(labels_today)
        for la in unique_labels:
            if la <= 0:
                continue

            mask0 = labels_today == la
            indices = np.argwhere(mask0)
            values = lwa_today[mask0]
            max_idx = np.argmax(values)
            centerlatindex, centerlonindex = indices[max_idx]

            if rgmask[centerlatindex, centerlonindex] == 1:
                k += 1
                regionLabels[day][mask0] = k
                regionlabelList.append(mask0)
                regionlabelcenterLat.append(centerlatindex)
                regionlabelcenterLon.append(centerlonindex)
                regionlabelmaxima.append(np.nanmax(values))

    np.save(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelIndexArr_6Hourly_{rgname}.npy', regionLabels)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelList_6Hourly_{rgname}', "wb") as f:
        pickle.dump(regionlabelList, f)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelCenterLat_6Hourly_{rgname}', "wb") as f:
        pickle.dump(regionlabelcenterLat, f)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelCenterLon_6Hourly_{rgname}_lon', "wb") as f:
        pickle.dump(regionlabelcenterLon, f)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelMaxima_6Hourly_{rgname}', "wb") as f:
        pickle.dump(regionlabelmaxima, f)
    print(f' {rgname} region label saved', flush=True)

    regionLabels = np.load(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelIndexArr_6Hourly_{rgname}.npy')
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelList_6Hourly_{rgname}', "rb") as f:
        regionlabelList = pickle.load(f)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelCenterLat_6Hourly_{rgname}', "rb") as f:
        regionlabelcenterLat = pickle.load(f)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelCenterLon_6Hourly_{rgname}_lon', "rb") as f:
        regionlabelcenterLon = pickle.load(f)
    with open(f'/scratch/bell/hu1029/LGHW/Blk_regionLabelMaxima_6Hourly_{rgname}', "rb") as f:
        regionlabelmaxima = pickle.load(f)
    print(f' {rgname} region label loaded', flush=True)

    # %% 04 find the local maxima of each blocking's first day

    for typeid in [1, 2, 3]:

        blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
        blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
        blockingEidArr = np.repeat(blockingEidArr, 4, axis=0)

        with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
            BlkEventlist = pickle.load(f)
        # get the first day date of each blocking event, 6-hourly
        with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp: 
            firstday_Date = pickle.load(fp) # hourly list
        firstblkIndex = [(timei.index(i)) for i in firstday_Date] # transfer to location index, 6-hourly
        print('blocking first day index:', firstblkIndex, flush=True)

        Blk1stMaxList = []
        centerlatList = []
        centerlonList = []
        for k,dayi in enumerate(firstblkIndex):

            print(f'blocking id: {BlkEventlist[k]}', flush=True)
            print(f'first day index: {dayi}', flush=True)
            blkarr = blockingEidArr[dayi,:,:] # transfer to boolean array
            mask0 = blkarr == BlkEventlist[k] # get the blocking event mask
            if np.sum(mask0) == 0:
                print(f'warning! No blocking event found for id {BlkEventlist[k]} on {firstday_Date[k]}', flush=True)
            # get the local maxima location
            LWA_avg = np.nanmean(LWA_td[dayi:dayi+4,:,:],axis=0)
            center0 = np.unravel_index(np.argmax(LWA_avg * mask0, axis=None), LWA_avg.shape)
            centerlatindex = center0[0]  # the latitude index of the center
            centerlonindex = center0[1]  # the longitude index of the center
            LocalMaxima = np.nanmax(LWA_avg * mask0)
            print(f'local maxima: {LocalMaxima}', flush=True)
            Blk1stMaxList.append(LocalMaxima)
            centerlatList.append(centerlatindex)
            centerlonList.append(centerlonindex)

            # # check the situation for small local maxima
            # if LocalMaxima < 0.9:
            #     blklen = 4*5
            #     dd = dayi
            #     for j in np.arange(blklen):
            #         dayi = dd + j

            #         blkarr = blockingEidArr[dayi,:,:] # transfer to boolean array
            #         mask0 = blkarr == BlkEventlist[k] # get the blocking event mask

            #         print(f'Warning! Local maxima {LocalMaxima} is smaller than 0.9 for blocking id {BlkEventlist[k]} on {firstday_Date[k]}', flush=True)
            #         print('center lat index:', centerlatindex, 'center lon index:', centerlonindex, flush=True)
            #         print('center lat:', latLWA[centerlatindex], 'center lon:', lonLWA[centerlonindex], flush=True)

            #         fig, ax, cf = create_Map(lonLWA,latLWA,LWA_td[dayi,:,:],fill=True,fig=None,
            #                                 leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
            #                                     minv=0, maxv=np.nanmax(LWA_td[dayi,:,:]), interv=11, figsize=(12,5),
            #                                     centralLon=loncenter, colr='PuBu', extend='max',title=f'LWA on {timei[dayi]}')
            #         ax.contour(lonLWA, latLWA, mask0.astype(float), levels=[0.5],
            #                             colors='darkblue', linewidths=2, transform=ccrs.PlateCarree())
            #         ax.contour(lonLWA, latLWA, WEvent[dayi,:,:], levels=[0.9],
            #                             colors='red', linewidths=1, transform=ccrs.PlateCarree())
                    
            #         addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
            #         plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

            #         plt.show()
            #         plt.savefig(f'test_localMaxima_blkevent{BlkEventlist[k]}_day{dayi}.png', bbox_inches='tight', dpi=300)
            #         plt.close()            

        print('minimum local maxima:', np.nanmin(Blk1stMaxList), flush=True)
        # blocking id: 519
        # first day index: 20728
        # local maxima: 0.8526469457412746
        
        plt.figure(figsize=(8, 6))
        sns.kdeplot(regionlabelmaxima, label='regionlabelmaxima', color='blue', linewidth=2)
        sns.kdeplot(Blk1stMaxList, label='Blk1stMaxList', color='red', linewidth=2)
        plt.title(f'Maximum LWA PDF ({rgname}) - {ss} - Type {typeid}')
        plt.xlabel('LWA (m²)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'Blocking1stLWACompare_PDF_Type{typeid}_{rgname}_{ss}.png', dpi=300)
        plt.close()

        print('done')
        
