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
cyc = "CC"

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

# get the event's label
if rgname == "SP":
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily_SH", "rb") as fp:
        Blocking_diversity_label = pickle.load(fp)
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily_SH", "rb") as fp:
        Blocking_diversity_date = pickle.load(fp)
else:
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily", "rb") as fp:
        Blocking_diversity_label = pickle.load(fp)
    with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
        Blocking_diversity_date = pickle.load(fp)
Blocking_diversity_label = Blocking_diversity_label[typeid-1] # get the blocking event list for the typeid
Blocking_diversity_date = Blocking_diversity_date[typeid-1] # get the blocking event date for the typeid

blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
# transfer to 6-hourly
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
blockingEidArr = np.repeat(blockingEidArr, 4, axis=0) # turn daily LWA to 6-hourly (simply repeat the daily value 4 times)

# read in the event's LWA list
with open(f"/scratch/bell/hu1029/LGHW/BlockEventDailyLWAList_1979_2021_Type{typeid}_{rgname}_{ss}.pkl", "rb") as f:
    BlkeventLWA = pickle.load(f) 

# the id of each blocking event (not all events! just the events within the target region)
with open(f'/scratch/bell/hu1029/LGHW/BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
    blkEindex = pickle.load(f)

# read in the track's interaction information
if rgname == "SP":
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021_SH.pkl', 'rb') as file:
        LWAtrack = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks_SH.pkl', 'rb') as file:
        track_data = pickle.load(file)
else:
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}TrackLWA_1979_2021.pkl', 'rb') as file:
        LWAtrack = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/{cyc}Zanom_allyearTracks.pkl', 'rb') as file:
        track_data = pickle.load(file)

EddyNumber = np.load(f'/scratch/bell/hu1029/LGHW/BlockingType{typeid}_EventEddyNumber_1979_2021_{rgname}_{ss}_{cyc}.npy')
eddyBlockIndex = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy')
    

# %% 02 test plotting for one event ------------------------
interactingBlkID = eddyBlockIndex
interactingBlkID = interactingBlkID[np.where(interactingBlkID >= 0)]
unique_ids, counts = np.unique(interactingBlkID, return_counts=True) # counts is the number of eddies in each block (unique_ids)
count_vals, count_freq = np.unique(counts, return_counts=True) # count_vals is the number of eddies in each block, count_freq is the frequency of each count value
for c, freq in zip(count_vals, count_freq):
    print(f"number of blocking that have {c} times eddy interaction: {freq}", flush=True)
idx = np.where(counts == 3)[0][3]
testblkid = unique_ids[idx] # the blocking with 3 eddies
print(f'Test blocking id: {testblkid}', flush=True)

# find the 3d slice in the BlockingEIndex array where the first dimension matches the target_value
maskday = np.any(blockingEidArr == testblkid, axis=(1, 2))  #shape=(time,) 
print('event length:', np.nansum(maskday), flush=True)
BlkTargetArr = blockingEidArr[maskday]
BlkTargetArr = BlkTargetArr>=0 # convert to boolean, for making the contour line
# get the target LWA 
LWATargetArr = LWA_td[maskday, :, :] # get the LWA for the target date
timeBlock = np.array(timei)[np.where(maskday)[0]] # the time Dates for the target blocking event

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
for i in range(np.shape(LWATargetArr)[0]):
    print('Plotting day:', i, flush=True)
    fig, ax, cf = create_Map(lonLWA,latLWA,LWATargetArr[i,:,:],fill=True,fig=None,
                             leftlon=lon_min1, rightlon=lon_max1, lowerlat=lat_min1, upperlat=lat_max1,
                                minv=0, maxv=np.nanmax(LWATargetArr), interv=11, figsize=(12,5),
                                centralLon=loncenter, colr='PuBu', extend='max',title=f'Blocking LWA on {timeBlock[i]}',)
    ax.contour(lonLWA, latLWA, BlkTargetArr[i,:,:].astype(float), levels=[0.5], 
                     colors='darkblue', linewidths=1.5, transform=ccrs.PlateCarree())
    
    addSegments(ax,[(lon_min,lat_min),(lon_max,lat_min),(lon_max,lat_max),(lon_min,lat_max),(lon_min,lat_min)],colr='black',linewidth=1)  
    plt.colorbar(cf,ax=ax,orientation='horizontal',label='LWA',fraction=0.04, pad=0.1)

    for track_id, points in TargetperiodTrackList[i]:
        addSegments(ax,points,colr='orangered',linewidth=2,alpha=0.2)
        if(len(points) > 0):
            end_lon, end_lat = points[-1]
            ax.plot(end_lon, end_lat, marker='x', color='orangered', markersize=6, 
                    transform=ccrs.PlateCarree())

    plt.show()
    plt.savefig(f'./AnimationMap/Blocking_LWA_EventID{testblkid}_CC_Day_{i}.png', bbox_inches='tight', dpi=300)
    plt.close()

# make animation ---------------------
image_folder = './AnimationMap'  # folder to save images
# get all the image files in the folder and sort them
frames = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.startswith(f'Blocking_LWA_EventID{testblkid}_CC') and f.endswith('.png')
], key=lambda f: int(f.split('_')[-1].split('.')[0]))
print(frames)
# get all the images
gif_filename = f"./AnimationMap/Animation_EventID{testblkid}_CC.gif"
with imageio.get_writer(gif_filename, mode="I", fps=6) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

print(f"GIF saved as {gif_filename}")
