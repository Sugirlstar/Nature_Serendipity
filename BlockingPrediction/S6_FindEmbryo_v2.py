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

# read in the LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

# # %% 01 get the LWA events ------------------------
# # read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)

rgname = 'ATL'
lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
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
lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)

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


Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
# get the wave event
T = LWA_td.shape[0]
print('Total number of timesteps:', T, flush=True)

T4 = T // 4
LWA_Z = LWA_td[:T4*4]
LWA_Z = LWA_Z.reshape(T4, 4, len(latLWA), len(lonLWA)).mean(axis=1) # daily mean of LWA

nlon = len(lonLWA)
nlat = len(latLWA)
nday = np.shape(LWA_Z)[0] # the number of days
# LWA_max_lon = np.zeros((nlon*nday))
# for t in np.arange(nday):    
#     for lo in np.arange(nlon):
#         LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
# Thresh = np.median(LWA_max_lon[:])
# print('Threshold:', Thresh)
# ### Wave Event ###
# WEvent = np.zeros((nday,nlat,nlon),dtype='uint8') 
# WEvent[LWA_Z>Thresh] = 1                    # Wave event daily

# ### connected component-labeling algorithm ###
# num_labels = np.zeros(nday)
# labels = np.zeros((nday,nlat,nlon))
# for d in np.arange(nday):
#     num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WEvent[d,:,:], connectivity=4)

# ####### connect the label around 180, since they are labeled separately ########
# ####### but actually they should belong to the same label  ########
# labels_new = copy.copy(labels)
# for d in np.arange(nday):
#     if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at either -180 or 179.375, then WEvent don't need to do any connection
#         continue
            
#     column_0 = np.zeros((nlat,3))       ## WEvent assume there are at most three wave events at column 0 (-180) (actuaaly most of the time there is just one)
#     column_end = np.zeros((nlat,3))
#     label_0 = np.zeros(3)
#     label_end = np.zeros(3)
    
#     ## Get the wave event at column 0 (0) ##
#     start_lat0 = 0
#     for i in np.arange(3):
#         for la in np.arange(start_lat0, nlat):
#             if labels_new[d,la,0]==0:
#                 continue
#             if labels_new[d,la,0]!=0:
#                 label_0[i]=labels_new[d,la,0]
#                 column_0[la,i]=labels_new[d,la,0]
#                 if labels_new[d,la+1,0]!=0:
#                     continue
#                 if labels_new[d,la+1,0]==0:
#                     start_lat0 = la+1
#                     break 

#         ## Get the wave event at column -1 (359) ## 
#         start_lat1 = 0
#         for j in np.arange(3):
#             for la in np.arange(start_lat1, nlat):
#                 if labels_new[d,la,-1]==0:
#                     continue
#                 if labels_new[d,la,-1]!=0:
#                     label_end[j]=labels_new[d,la,-1]
#                     column_end[la,j]=labels_new[d,la,-1]
#                     if labels_new[d,la+1,-1]!=0:
#                         continue
#                     if labels_new[d,la+1,-1]==0:
#                         start_lat1 = la+1
#                         break                       
#             ## Compare the two cloumns at 0 and 359, and connect the label if the two are indeed connected
#             if (column_end[:,i]*column_0[:,j]).mean() == 0:
#                 continue                
#             if (column_end*column_0).mean() != 0:
#                 num_labels[d]-=1
#                 if label_0[i] < label_end[j]:
#                     labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
#                     labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
#                 if label_0[i] > label_end[j]:
#                     labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
#                     labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]

# print('labels -> labels_new, num_labels -> num_labels', flush=True)

# np.save(f'/scratch/bell/hu1029/LGHW/Blk_LWAconnectedDailyLabels{HMi}.npy', labels_new)
# print('connected labels saved', flush=True)

# labels_new_origin = copy.copy(labels_new) # save the original labels_new for later use

# # %% 02 get the blocking embryo ------------------------

# qualified_tracks = []
# qualified_masks = []
# qualified_dateindices = []
# for t0 in np.arange(nday-2): # for each day, find the potential wave event and the consecutive clusters
    
#     print(' Processing day:', t0, flush=True)

#     for label0 in np.unique(labels_new[t0]):
#         if label0 == 0:  
#             continue

#         track = [(t0, label0)]
#         mask0 = labels_new[t0] == label0
#         masks = [mask0]
#         dateindices = [t0]

#         center0 = np.unravel_index(np.argmax(LWA_Z[d,:,:] * mask0, axis=None), LWA_Z[d,:,:].shape)
#         mask0width = get_lon_width(mask0, nlon)
#         width_count = 1 if mask0width > 15 else 0

#         curr_center = center0
#         for t_next in range(t0 + 1, nday):
#             found_match = False
#             candidates = []
#             for label_next in np.unique(labels_new[t_next]):
                
#                 if label_next == 0:
#                     continue
#                 mask_next = labels_new[t_next] == label_next
#                 center_next = np.unravel_index(np.argmax(LWA_Z[d,:,:] * mask_next, axis=None), LWA_Z[d,:,:].shape)

#                 # displacement check        
#                 dy = abs(center_next[0] - curr_center[0])
#                 dx = abs(center_next[1] - curr_center[1]) 

#                 if dx <= 18 and dy <= 13.5:
#                     dist = float(np.sqrt(dy**2 + dx**2))
#                     candidates.append((label_next, center_next, dist, mask_next))

#             if candidates:
#                 best = min(candidates, key=lambda x: x[2])  
#                 label_next, center_next, _, mask_next = best
#                 track.append((t_next, label_next))
#                 masks.append(mask_next)
#                 dateindices.append(t_next)
#                 curr_center = center_next
#                 if get_lon_width(mask_next, nlon) > 15:
#                     width_count += 1
#                 found_match = True
#             else:
#                 break

#         if len(track) >= 2 and width_count >= 2:
#             qualified_tracks.append(track)
#             qualified_masks.append(np.stack(masks))
#             qualified_dateindices.append(dateindices)
#             print(f'Qualified track found on day {t0}:', flush=True)
#         for tt, ll in track:
#             labels_new[tt][labels_new[tt] == int(ll)] = 0

# print(qualified_tracks[0], flush=True)
# print([len(track) for track in qualified_tracks], flush=True)
# print("Shape of first mask array:", qualified_masks[0].shape)

# # get the embryo events
# print('Number of qualified tracks:', len(qualified_tracks), flush=True)
# embryoArr = np.full_like(LWA_Z, -1, dtype=np.int64)  # initialize the embryo array - daily arr
# for trackid, track in enumerate(qualified_tracks):
#     for tt, ll in track:
#         embryoArr[tt][labels_new_origin[tt] == int(ll)] = trackid

# np.save(f'/scratch/bell/hu1029/LGHW/EmbryoIndexArrayDaily_v2.npy', embryoArr)
# print(f'Embryo array saved for {rgname}', flush=True)
# pickle.dump(qualified_masks, open(f'/scratch/bell/hu1029/LGHW/EmbryoTrackedLabels.pkl', 'wb'))  # save the qualified tracks
# pickle.dump(qualified_dateindices, open(f'/scratch/bell/hu1029/LGHW/EmbryoTrackedDates.pkl', 'wb'))  # save the qualified masks

# %% 03 filter the embryo events and blocking events ------------------------
ss = 'ALL'
rgname = 'ATL'

embryoArr = np.load(f'/scratch/bell/hu1029/LGHW/EmbryoIndexArrayDaily_v2.npy') # the embryo event index array, daily value
qualified_masks = pickle.load(open(f'/scratch/bell/hu1029/LGHW/EmbryoTrackedLabels.pkl', 'rb'))  # load the qualified tracks
qualified_dateindices = pickle.load(open(f'/scratch/bell/hu1029/LGHW/EmbryoTrackedDates.pkl', 'rb'))  # load the qualified date indices

# for each embryo event, check if they are within the target region
regionTarget_embryosLabels = []  # the target region indices
regionTarget_embryosDates = []  # the target region date indices
for k in np.arange(len(qualified_masks)):

    emeventlabel = qualified_masks[k]  # the embryo event label
    emeventdate = qualified_dateindices[k]  # the embryo event date indices
    emeventRegionCheck = rgmask * emeventlabel
    emeventRegionChecksum = np.nansum(emeventRegionCheck, axis=(1, 2))  # sum over lat and lon
    if np.any(emeventRegionChecksum > 0):  # if there is any value greater than 0, it means the embryo event is within the target region
        print(f'Embryo event {k} is within the target region', flush=True)
        regionTarget_embryosLabels.append(emeventlabel)  # add the embryo event label to the target region list
        regionTarget_embryosDates.append(emeventdate)  # add the embryo event date indices to the target region list

print(f'Total number of embryo events within the target region: {len(regionTarget_embryosLabels)}', flush=True)


# %% start to filter the embryo events, including both 3 types of blocking events -------------------------

connectedblknumList = []
relatedblknumList = []
connectedblkEmbryoIDList = []
relatedblkEmbryoIDList = []
excludedEmbryoIDList = []

for _,typeid in enumerate([1,2,3]):
        
    blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
    blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
    with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
        BlkEventlist = pickle.load(f)
    # get the first day of each blocking event, 6-hourly
    with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
        firstday_Date = pickle.load(fp)
    firstblkIndex = [(timei.index(i))//4 for i in firstday_Date]  # the first blocking day index in the timei list, transfer to daily locations
    print('First blocking day index: ', firstblkIndex, flush=True)
    blkid_to_firstday = {bid: firstblkIndex[i] for i, bid in enumerate(BlkEventlist)}

    # for each embryo event, compare it with the real blocking events
    # if it = blocking event, delete it
    # if it is later than the blocking event (after 1 days), delete it
    # if it is before and have a overlapping with the blocking event, keep it as an embryo event

    for k in np.arange(len(regionTarget_embryosLabels)):

        print(f'checkpoint1: Processing embryo event {k} for type {typeid}', flush=True)
        emeventlabel = regionTarget_embryosLabels[k]  # the embryo event label
        emeventdate = regionTarget_embryosDates[k]  # the embryo event date indices
        print(f'Embryo event date indices: {emeventdate}', flush=True)

        # get the blocking event id arr for the same period
        blkarr = blockingEidArr[emeventdate, :, :]  # the blocking event id array for the same period
        blkarr = blkarr>=0

        if k == 0:
            for i,t in enumerate(emeventdate):
                # make the plot to check the sanity
                fig, ax, cf = create_Map(lonLWA,latLWA,LWA_Z[t,:,:],fill=True,fig=None,
                                        leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                            minv=0, maxv=np.nanmax(LWA_Z[t,:,:]), interv=11, figsize=(12,5),
                                            centralLon=loncenter, colr='PuBu', extend='max',title=f'{Date[t]}')
                ax.contour(lonLWA, latLWA, emeventlabel[i,:,:].astype(float), levels=[0.5], 
                                colors='darkred', linewidths=2, transform=ccrs.PlateCarree())
                ax.contour(lonLWA, latLWA, blkarr[i,:,:].astype(float), levels=[0.5], 
                                colors='darkblue', linewidths=1, transform=ccrs.PlateCarree())
            
                addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
                plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

                plt.show()
                plt.savefig(f'FindEmbryo_v2_test{k}_d{i}.png', bbox_inches='tight', dpi=300)
                plt.close()

        if(np.nansum(blkarr) > 0):
            for i,t in enumerate(emeventdate):
                # make the plot to check the sanity
                fig, ax, cf = create_Map(lonLWA,latLWA,LWA_Z[t,:,:],fill=True,fig=None,
                                        leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                            minv=0, maxv=np.nanmax(LWA_Z[t,:,:]), interv=11, figsize=(12,5),
                                            centralLon=loncenter, colr='PuBu', extend='max',title=f'{Date[t]}')
                ax.contour(lonLWA, latLWA, emeventlabel[i,:,:].astype(float), levels=[0.5], 
                                colors='darkred', linewidths=2, transform=ccrs.PlateCarree())
                ax.contour(lonLWA, latLWA, blkarr[i,:,:].astype(float), levels=[0.5], 
                                colors='darkblue', linewidths=1, transform=ccrs.PlateCarree())
            
                addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
                plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

                plt.show()
                plt.savefig(f'FindEmbryo_v2_test{k}_d{i}.png', bbox_inches='tight', dpi=300)
                plt.close()
        
                print(checkpoint3)
    

 











    





    
#     # start to filter
#     connectedblknum = [0] * len(embryovalues)  # the number of blocking events after the embryo event
#     relatedblknum = [0] * len(embryovalues)  # the number of blocking events related to the embryo event
#     connectedblkEmbryoID = [] 
#     relatedblkEmbryoID = []
#     excludedEmbryoID = []
#     relatedblkID = [[] for _ in range(len(embryovalues))] # the same length as total embryovalues

#     for ki in embryovalues:

#         emdayi = np.array(sorted(set(embryo_id_to_days[ki])))
#         embryoslice = embryoArr[emdayi] >= 0  
#         print(f'Embryo ID {ki}, days: {emdayi}', flush=True)
#         first_embryo_day = emdayi[0]  # the first day of the embryo event
#         blkvalue = blockingEidArr[emdayi, :, :]
#         blkvalue = blkvalue.astype(float)
#         blkvalue[~embryoslice.astype(bool)] = np.nan  # get the blocking event id for the current embryo id
#         flat_blkvalue = blkvalue[blkvalue>=0]

#         blockingids = np.unique(flat_blkvalue) # the blocking's id

#         if len(blockingids) == 0:
#             emdayafter = emdayi[-1] + 1
#             emdayafterwindow = np.arange(emdayafter, min(emdayafter + 3, nday))  # the next 3 days after the last embryo day
#             embryoslice2 = embryoArr[emdayafterwindow, :, :] >= 0  # get the embryo slice for the next 3 days
#             # find if there are any blocking events in the next 3 days
#             blkvalue = blockingEidArr[emdayafterwindow, :, :]
#             blkvalue = blkvalue.astype(float)
#             blkvalue[~embryoslice2.astype(bool)] = np.nan  # get the blocking event id for the current embryo id
#             flat_blkvalue = blkvalue[blkvalue>=0]
#             afterblockingids = np.unique(flat_blkvalue) # the blocking's id
#             if len(afterblockingids)>0:
#                 print(f'    Embryo {ki} has a related blocking in the next 3 days: {afterblockingids}', flush=True)
#                 relatedblknum[ki] = len(afterblockingids)  # the number of blocking events after the embryo event
#                 relatedblkEmbryoID.append(ki)  # add the embryo id to the before blocking embryo id list
#                 relatedblkID[ki] = afterblockingids.tolist()  # add the blocking ids to the related blocking id list
#         else:
#             if all(first_embryo_day < blkid_to_firstday[int(i)] for i in blockingids):
#                 print(f'    Embryo {ki} has connected blockings, all starting after embryo', flush=True)
#                 connectedblknum[ki] = len(blockingids)  
#                 connectedblkEmbryoID.append(ki)
#                 relatedblkID[ki] = blockingids.tolist()  # add the blocking ids to the related blocking id list
#             else:
#                 print(f'    Embryo {ki} is part of at least a blocking', flush=True)
#                 excludedEmbryoID.append(ki)

#     connectedblknum = np.array(connectedblknum)
#     connectedblknum = connectedblknum.tolist()  # convert to list 
#     relatedblknum = np.array(relatedblknum)
#     relatedblknum = relatedblknum.tolist()  # convert to list

#     print(f'{typeid}_{rgname}_{ss}, Total Embryo event : ', len(embryovalues), flush=True)
#     print(f'{typeid}_{rgname}_{ss}, Eexcluded Embryo event: ', len(excludedEmbryoID), flush=True)
#     print(f'{typeid}_{rgname}_{ss}, embyro with blocking:', len(connectedblkEmbryoID), flush=True)
#     print(f'{typeid}_{rgname}_{ss}, embyro with after blocking:', len(relatedblkEmbryoID), flush=True)
#     print(f'{typeid}_{rgname}_{ss}, excluded ID: ', excludedEmbryoID, flush=True)
#     print(f'{typeid}_{rgname}_{ss}, ----------------------------------------------- ', flush=True)

#     flat_relatedblkID = np.array([x for sublist in relatedblkID for x in sublist])
#     unique_blockingID = np.unique(flat_relatedblkID)
#     blockingids = np.array(BlkEventlist)
#     missing_ids = np.setdiff1d(blockingids, unique_blockingID)
#     present_ids = np.intersect1d(blockingids, unique_blockingID)

#     print(f'* {rgname}_{typeid}: , Blocking with embryo length: {len(present_ids)}', flush=True)
#     print(f'* {rgname}_{typeid}: , Blocking without embryo length: {len(missing_ids)}', flush=True)
#     print(f'* {rgname}_{typeid}: , Blocking without embryo percentage: {len(missing_ids)/len(blockingids)}', flush=True)
#     print(f'* {rgname}_{typeid}: , Blocking without embryo IDs: {missing_ids}', flush=True)

#     # save the related blocking event numbers
#     successNum = np.array(connectedblknum) + np.array(relatedblknum)
#     np.save(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_BlkType{typeid}_{rgname}.npy', successNum)
#     with open(f'/scratch/bell/hu1029/LGHW/embryo_relatedblkID_BlkType{typeid}_{rgname}', "wb") as f:
#         pickle.dump(relatedblkID, f)
#     print('related Blk IDs: \n',relatedblkID, flush=True)

#     connectedblknumList.append(connectedblknum)
#     relatedblknumList.append(relatedblknum)
#     connectedblkEmbryoIDList.append(connectedblkEmbryoID)
#     relatedblkEmbryoIDList.append(relatedblkEmbryoID)
#     excludedEmbryoIDList.append(excludedEmbryoID)

# excludedEmbryoID_flat = [item for sublist in excludedEmbryoIDList for item in sublist]
# ExcludedEmbryoID = np.unique(excludedEmbryoID_flat)
# connectedBlkNumbers = np.sum(connectedblknumList, axis=0)
# relatedBlkNumbers = np.sum(relatedblknumList, axis=0)
# totalBlkedNumbers = np.array(connectedBlkNumbers) + np.array(relatedBlkNumbers) # the number of blocking events related to each embryo event
# totalBlkedNumbers = np.array(totalBlkedNumbers)  # convert to numpy array
# totalBlkedNumbers[ExcludedEmbryoID] = 0  # set the excluded embryo id's blocking number to 0

# SuccssEmbryoID = np.where(totalBlkedNumbers > 0)[0]  # the embryo ids that have blocking events (in global location)


# # get the real embryo ids:
# RealEmbryoIDs = [eid for eid in embryovalues if eid not in ExcludedEmbryoID]
# print(f'Real Embryo length: {len(RealEmbryoIDs)}', flush=True)
# print(f'Successful Embryo length: {len(SuccssEmbryoID)}', flush=True)

# np.save(f'/scratch/bell/hu1029/LGHW/embryo_RealEmbryoIDs_{rgname}.npy', RealEmbryoIDs) # the location of the real embryo ids in all embryovalues
# np.save(f'/scratch/bell/hu1029/LGHW/embryo_SuccessNumbersEachEmbryo_{rgname}.npy', totalBlkedNumbers)
# np.save(f'/scratch/bell/hu1029/LGHW/embryo_SuccssEmbryoID_{rgname}.npy', SuccssEmbryoID) # the location of the successful embryo ids in all embryovalues

# print(f'Saved for {rgname} in {ss}', flush=True)
