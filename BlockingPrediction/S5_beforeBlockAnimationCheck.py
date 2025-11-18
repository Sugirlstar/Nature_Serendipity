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

# get the wave event
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
Duration = 5 
print('Threshold:', Thresh)
### Wave Event ###
WEvent = np.zeros((nday,nlat,nlon),dtype='uint8') 
WEvent[LWA_Z>Thresh] = 1                    # Wave event daily
WEvent = np.repeat(WEvent, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

# blocking event id array
blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
# transfer to 6-hourly
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

# the id of each blocking event (not all events! just the events within the target region)
with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
    blkEindex = pickle.load(f)

# the first day of each blocking event
with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "rb") as fp:
    firstday_Date = pickle.load(fp)
firstblkIndex = [(timei.index(i)) for i in firstday_Date]  # the first blocking day index in the timei list
print('First blocking day index: ', firstblkIndex, flush=True)
blkid_to_firstday = {bid: firstblkIndex[i] for i, bid in enumerate(blkEindex)}

# read in the track's interaction information
if rgname == "SP":
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_SH.pkl', 'rb') as file:
        track_data = pickle.load(file)
else:
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks.pkl', 'rb') as file:
        track_data = pickle.load(file)

# the blocking event index each eddy interacts with (-1 means no interaction)
eddyBlockIndex = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')

if rgname == "SP":
    HMi = '_SH'
else:
    HMi = ''
trackPoints_array = np.load(f'/scratch/bell/hu1029/LGHW/{cyc}trackPoints_array{HMi}.npy') # all track points
dstrack = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
tracklon = np.array(ds['lon'])
tracklat = np.array(ds['lat'])
tracklat = np.flip(tracklat)
if rgname == "SP":
    tracklat = tracklat[0:(findClosest(0,tracklat)+1)]
else:
    tracklat = tracklat[(findClosest(0,tracklat)+1):len(tracklat)]
print('tracklat:', tracklat, flush=True) # make sure it's in ascending order (from south to north)

lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)

# # %% 02 test plotting for one event ------------------------
# interactingBlkID = eddyBlockIndex
# interactingBlkID = interactingBlkID[np.where(interactingBlkID >= 0)]
# unique_ids, counts = np.unique(interactingBlkID, return_counts=True) # counts is the number of eddies in each block (unique_ids)
# count_vals, count_freq = np.unique(counts, return_counts=True) # count_vals is the number of eddies in each block, count_freq is the frequency of each count value

# for countsnum in [1,2,3]:
#     for k in [2,4,6,8,10]: # 
#         idx = np.where(counts == countsnum)[0][k]
#         testblkid = unique_ids[idx] # the blocking with 3 eddies
#         print(f'Test blocking id: {testblkid}', flush=True)

#         # find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
#         maskday = np.any(blockingEidArr == testblkid, axis=(1, 2))  #shape=(time,) 
#         # get the target LWA 
#         print('its first blocking date:', timei[np.where(maskday)[0][0]], flush=True)
#         nextdayid = np.where(maskday)[0][0] - 12 # the next 4 day index
#         if nextdayid < 0: # if the next day is before the first day, then set it to 0
#             continue
#         next4dayid = np.where(maskday)[0][0] + 13 # the next 4 day index
        
#         maskday = np.zeros_like(maskday, dtype=bool) # create a mask for the target blocking event
#         maskday[nextdayid:next4dayid] = True # include the next day
#         timeBlock = np.array(timei)[np.where(maskday)[0]] # the time Dates for the target blocking event
#         # get the target Wave event bool array
#         WEventTargetArr = WEvent[maskday, :, :] # get the Wave event for the target date

#         print('event length:', np.nansum(maskday), flush=True)
#         BlkTargetArr = blockingEidArr[maskday]
#         print('blockingArr values:', np.unique(BlkTargetArr), flush=True)

#         BlkTargetArr = BlkTargetArr>=0 # convert to boolean, for making the contour line

#         LWATargetArr = LWA_td[maskday, :, :] # get the LWA for the target date
#         trackpointArr = trackPoints_array[maskday, :, :] # get the track point for the target date


#         # find the eddy indices for the target blocking event
#         eddyindices = np.where(eddyBlockIndex == testblkid)[0] # the eddy indices for the target blocking event
#         targetTracks = [track_data[i] for i in eddyindices] # the track data for the target blocking event

#         daylen = np.shape(LWATargetArr)[0]
#         TargetperiodTrackList = [] # a list to store the track point tracks
#         for i in range(len(timeBlock)):  
#             theday = timeBlock[i] # the target day
#             track_points = [
#             (index, [(lon, lat) for timeid, lon, lat in points if timeid <= theday ]) 
#             for index, points in targetTracks
#             ]
#             TargetperiodTrackList.append(track_points)
#             print(track_points)

#         print(TargetperiodTrackList[0])
#         print(TargetperiodTrackList[1])
#         print(TargetperiodTrackList[2])         


#         # plot1: the map of the LWA
#         lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
#         lat_min1, lat_max1, lon_min1, lon_max1, loncenter = PlotBoundary(rgname)
#         for i in range(np.shape(LWATargetArr)[0]):
            
#             dayi = i-12
#             if dayi >= 0:
#                 titletext = f'Blocking LWA on {timeBlock[i]}'
#             else:
#                 titletext = f'LWA on {timeBlock[i]})'
            
#             print('Plotting day:', i, flush=True)
#             fig, ax, cf = create_Map(lonLWA,latLWA,LWATargetArr[i,:,:],fill=True,fig=None,
#                                     leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
#                                         minv=0, maxv=np.nanmax(LWATargetArr), interv=11, figsize=(12,5),
#                                         centralLon=loncenter, colr='PuBu', extend='max',title=titletext,)
#             ax.contour(lonLWA, latLWA, WEventTargetArr[i,:,:].astype(float), levels=[0.5], 
#                             colors='darkred', linewidths=1.5, transform=ccrs.PlateCarree())
#             ax.contour(lonLWA, latLWA, BlkTargetArr[i,:,:].astype(float), levels=[0.5], 
#                             colors='darkblue', linewidths=2, transform=ccrs.PlateCarree())
            
#             addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
#             plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

#             for track_id, points in TargetperiodTrackList[i]:
#                 addSegments(ax,points,colr='orangered',linewidth=2,alpha=0.4)
#                 if(len(points) > 0):
#                     end_lon, end_lat = points[-1]
#                     ax.plot(end_lon, end_lat, marker='x', color='orangered', markersize=6, 
#                             transform=ccrs.PlateCarree())
            
#             AAlat,AAlon = np.where(trackpointArr[i]>=1)
#             AAlatvalue = tracklat[AAlat]
#             AAlonvalue = tracklon[AAlon]
#             ax.scatter(AAlonvalue, AAlatvalue, color='blue', s=20, label=f'{cyc} points', transform=ccrs.PlateCarree(),zorder=10)

#             plt.show()
#             plt.savefig(f'Blocking_LWA_EventID{testblkid}_Day_{dayi}.png', bbox_inches='tight', dpi=300)
#             plt.close()

#         # make animation ---------------------
#         image_folder = './'  # folder to save images
#         # get all the image files in the folder and sort them
#         frames = sorted([
#             os.path.join(image_folder, f)
#             for f in os.listdir(image_folder)
#             if f.startswith(f'Blocking_LWA_EventID{testblkid}_Day_') and f.endswith('.png')
#         ], key=lambda f: int(f.split('_')[-1].split('.')[0]))
#         print(frames)
#         # get all the images
#         gif_filename = f"./Animation_EventID{testblkid}_tracknumber{countsnum}.gif"
#         with imageio.get_writer(gif_filename, mode="I", fps=4) as writer:
#             for frame in frames:
#                 image = imageio.imread(frame)
#                 writer.append_data(image)

#         print(f"GIF saved as {gif_filename}")

#         for i in range(np.shape(LWATargetArr)[0]):
#             dayi = i-12
#             filename = f'./Blocking_LWA_EventID{testblkid}_Day_{dayi}.png'
#             if os.path.exists(filename):
#                 os.remove(filename)
#                 print(f'{filename} has been deleted.')
#             else:
#                 print(f'{filename} does not exist.')


# print('All selected blocking done!', flush=True)

# randomly plot the day i-15:i+15, showing WE, and the track points
np.random.seed(42)
dayrandoms = np.random.choice(np.arange(10, len(timei)), size=20, replace=False)

targeteventid = [17, 18, 19, 20, 29]

for i in targeteventid:

    firstday = blkid_to_firstday[int(i)]

    for j in range(-20,21):

        dayi = firstday+j
        blkarr = blockingEidArr[dayi,:,:]>=0
        print('Random day:', timei[dayi], flush=True)
        fig, ax, cf = create_Map(lonLWA,latLWA,LWA_td[dayi,:,:],fill=True,fig=None,
                                leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                    minv=0, maxv=np.nanmax(LWA_td[dayi,:,:]), interv=11, figsize=(12,5),
                                    centralLon=loncenter, colr='PuBu', extend='max',title=f'LWA on {timei[dayi]}')
        ax.contour(lonLWA, latLWA, WEvent[dayi,:,:].astype(float), levels=[0.5], 
                            colors='darkred', linewidths=1.5, transform=ccrs.PlateCarree())
        ax.contour(lonLWA, latLWA, blkarr.astype(float), levels=[0.5],
                            colors='darkblue', linewidths=2, transform=ccrs.PlateCarree())
        
        addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
        plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

        AAlat,AAlon = np.where(trackPoints_array[dayi]>=1)
        AAlatvalue = tracklat[AAlat]
        AAlonvalue = tracklon[AAlon]
        ax.scatter(AAlonvalue, AAlatvalue, color='blue', s=20, label=f'{cyc} points', transform=ccrs.PlateCarree(),zorder=10)

        plt.show()
        plt.savefig(f'test_blkevent{i}_Day_{j}.png', bbox_inches='tight', dpi=300)
        plt.close()

    # make animation ---------------------
    image_folder = './'  # folder to save images
    # get all the image files in the folder and sort them
    frames = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.startswith(f'test_blkevent{i}_Day_') and f.endswith('.png')
    ], key=lambda f: int(f.split('_')[-1].split('.')[0]))
    print(frames)
    # get all the images
    gif_filename = f"./Animation_test_blkevent{i}.gif"
    with imageio.get_writer(gif_filename, mode="I", fps=4) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    print(f"GIF saved as {gif_filename}")

    for j in range(-20,21):
        filename = f'./test_blkevent{i}_Day_{j}.png'
        if os.path.exists(filename):
            os.remove(filename)
            print(f'{filename} has been deleted.')
        else:
            print(f'{filename} does not exist.')

print('All random days done!', flush=True)