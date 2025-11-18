from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import glob
import copy
import pickle
import matplotlib.path as mpath
from netCDF4 import Dataset
import xarray as xr

# %% 00 function preparation --------------------------------

leadD = 4

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

#%% 01 read the data - time management
Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Month = Date0['date'].dt.month
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
# BAM dates
feb_29_ind = Date0[(Month == 2) & (Day == 29)].index
print(feb_29_ind, flush=True)

# read the BAM index
BI = np.load('/scratch/bell/hu1029/LGHW/SBAM_index_total_no_leap.npy')    ## This is full BAM index from 1979 to 2021, no leap years ###
print(BI.shape, flush=True)
print(BI[423],BI[424],BI[425],flush=True)  

# get back to the full time (with leap years)
BI_full = np.full(len(Date0), np.nan)
keep_idx = np.setdiff1d(np.arange(len(Date0)), feb_29_ind)
BI_full[keep_idx] = BI.squeeze()
BI_full_series = pd.Series(BI_full)
BI_full_interp = BI_full_series.interpolate(method='linear')
BI_full_interp = BI_full_interp.to_numpy()
print(BI_full_interp.shape)
print(BI_full_interp[423],BI_full_interp[424],BI_full_interp[425],flush=True)  
BI = BI_full_interp
BAMDate = Date0

# read the SH Blocking index 
B_freq_2d = np.load('/scratch/bell/hu1029/LGHW/B_freq_SH.npy')
B_freq_3d = np.load('/scratch/bell/hu1029/LGHW/B_freq2_SH.npy')

#%% identify BAM phases
BI_copy = copy.copy(BI)
BAM_event_all = []      ## To store the date or index with large pc1 value ##
BAM_event_BI_all = []   ## The value of the BI at that peaking date ##
number = 0
while 1:
    index = np.array(np.where( BI_copy==np.nanmax(BI_copy) ))[0][0]
    if BI_copy[index]>=BI[index+1] and BI[index]>=BI[index-1] and BI_copy[index]>=BI[index+2] and BI[index]>=BI[index-2]:        
        BAM_event_all.append(index)
        BAM_event_BI_all.append(BI[index])
        number+=1
        BI_copy[index-12:index+13] = np.nan
        if number > 550:
            break
    else:
        BI_copy[index]=np.nan

# remove the repeating values
seen = set()
BAM_event_all_unique = []
BAM_event_BI_all_unique = []
for idx, bi in zip(BAM_event_all, BAM_event_BI_all):
    if idx not in seen:
        seen.add(idx)
        BAM_event_all_unique.append(idx)
        BAM_event_BI_all_unique.append(bi)
BAM_event_all = BAM_event_all_unique
BAM_event_BI_all = BAM_event_BI_all_unique
print("BAM index:", BAM_event_all, flush=True)
print("BAM values:", BAM_event_BI_all, flush=True)
print("number of BAM events:", len(BAM_event_all), flush=True)

# get the high BAM states
T=1
n_BAM = len(BAM_event_all)
# B_B and B_LB is to store the BAM or Low BAM condition days
B_B = np.zeros((BI.shape[0],B_freq_3d.shape[2]))  ## This array is to store the blocking which falls into +-T day of BAM peaking date
B_LB = np.zeros((BI.shape[0],B_freq_3d.shape[2])) ## This array is to store the blocking whihc falls into +-T day of low BAM date
for i in np.arange(n_BAM):
    BAMloc = BAM_event_all[i]
    BAMday = BAMDate.iloc[BAMloc]
    B_B[BAMloc-T:BAMloc+T+1,:] = 1
    B_LB[BAMloc-12-T:BAMloc-12+T+1,:] = 1
print("B_B shape:", B_B.shape, flush=True) # high BAM center = 1, others = 0

#%% 02 read the blocking and track data (blocking grid flag)
# for each blocking event, it can be defined as related with AC/CC or not (1/0 on each grid point for the cluster)
# transfer to 2d array (time, lon)
typeid = 1
rgname = "SP"
ss = "ALL"
cyc = "AC"

# 02-1 read in the track's timesteps (6-hourly)
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
tracklon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
latNH = lat[0:findClosest(0,lat)+1]
print(latNH, flush=True)
# read in the track flag, and transfer to 2d array (time, lon)
trackPoints_array = np.load('/scratch/bell/hu1029/LGHW/ACtrackPoints_array_SH.npy')
print('original track point arr shape:', trackPoints_array.shape, flush=True) # 0 or 1
trackPoints_2D = np.any(trackPoints_array, axis=1).astype(int) # trackpoints that are 2D, with or without AC track points
print('trackPoints_2D shape:', trackPoints_2D.shape, flush=True) # (nday*4,

# 02-2 read in the blocking flag, and transfer to 2d array (time, lon)
blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
print('blockingEidArr shape:', blockingEidArr.shape, flush=True)
blockingEidArr = np.where(blockingEidArr != -1, 1, 0) # transfer to 0/1
blockingEidArr_2D = np.any(blockingEidArr, axis=1).astype(int)
print('blockingEidArr_2D shape:', blockingEidArr_2D.shape, flush=True) # (nday*4, 360), 0 or 1

# 02-3 read in the blocking lat and lon
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
    Blklat = lat[lat_mid:len(lat)]
else:
    Blklat = lat[0:lat_mid-1]
Blklat = np.flip(Blklat) # make it ascending order (from south to north)
print(Blklat, flush=True)
Blklon = lon 

# now we have trackPoints_2D, blockingEidArr_2D and BAM index (1D)
print('trackPoints_2D shape:', trackPoints_2D.shape, flush=True)
print('blockingEidArr_2D shape:', blockingEidArr_2D.shape, flush=True)
print('BAM index B_B shape:', B_B.shape, flush=True)

# 02-4 transfer BI and blockingEidArr_2D to the same time dimension as trackPoints_2D
# BI is daily, blockingEidArr_2D is daily, trackPoints_2D is 6-hourly
# so we need to repeat BI and blockingEidArr_2D 4 times to make it 6-hourly
print('----shape comply----', flush=True)
B_B_6hr = np.repeat(B_B, 4, axis=0)
B_LB_6hr = np.repeat(B_LB, 4, axis=0)
print('BAM index shape after reshaping:', B_B_6hr.shape, flush=True)
blockingEidArr_6hr = np.repeat(blockingEidArr_2D, 4, axis=0)
print('blockingEidArr_6hr shape after reshaping:', blockingEidArr_6hr.shape, flush=True)
print('trackPoints_2D shape:', trackPoints_2D.shape, flush=True)

#%% 03 check the probability of Blocking under HB or LB
nday = len(Date0)
blk_under_BAM = blockingEidArr_6hr * B_B_6hr
blk_under_LB = blockingEidArr_6hr * B_LB_6hr
lat_min, lat_max, lon_min, lon_max = Region_ERA(rgname)
ntime = nday*4 # 6-hourly
B_B_day = n_BAM*(2*T+1)*4 # number of days under high BAM condition, times 4 for 6-hourly
B_B_num = np.nansum(blk_under_BAM,axis=0)
B_LB_num = np.nansum(blk_under_LB,axis=0)
B_B_freq = (B_B_num/B_B_day)
B_LB_freq = (B_LB_num/B_B_day) # frequency of blocking under low BAM condition
B_freq_clima = np.nansum(blockingEidArr_6hr,axis=0)/ntime
T_freq_clima = np.nansum(trackPoints_2D,axis=0)/ntime

#%% 04 plot high/low BAM and climatology
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Blklon, B_B_freq, color='red', label='High BAM')
ax.plot(Blklon, B_LB_freq, color='blue', label='Low BAM')
ax.plot(Blklon, B_freq_clima, color='black', label='Climatology')
# target region shading
ax.axvspan(lon_min, lon_max, color='yellow', alpha=0.3, label=f'Region {lon_min}-{lon_max}')
# labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Blocking Frequency')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(f'BAM2D_BlockingFreq_AlongLon_{rgname}_Type{typeid}_{cyc}.png', dpi=300)

# %% 05 get the tracking days: those days where the region with at least 1 track
# for each lon grid point, check if there is any track point 4 days prior and 10 grids upstream
# if yes, then this day and grid is a tracking point

# 05-1 interpolate the trackPoints_2D to 360 lat grid points (from 512)
ntracktime, _ = trackPoints_2D.shape
lon_new = Blklon
trackPoints_2D_resampled = np.zeros((ntracktime, len(lon_new)))
for k in range(ntracktime):
    # get the 1D array for the current time step
    y_ind = np.where(trackPoints_2D[k, :] > 0)[0]
    print('y_ind:', y_ind, flush=True)
    y_ind_new = findClosest(tracklon[y_ind], lon_new)
    print('y_ind_new:', y_ind_new, flush=True)
    trackPoints_2D_resampled[k, y_ind_new] = 1

print('trackPoints_2D_resampled shape:', trackPoints_2D_resampled.shape, flush=True)
print('value compare:', np.where(trackPoints_2D[0,:]>0)[0], np.where(trackPoints_2D_resampled[0,:]>0)[0], flush=True)
print('lon compare:', tracklon[np.where(trackPoints_2D[0,:]>0)[0]], Blklon[np.where(trackPoints_2D_resampled[0,:]>0)[0]], flush=True)

# 05-2 create a mask for the track points, where there is at least one track point in the 4 days prior and 10 grids upstream
def kernel_left_up(m, n):
    H, W = 2*m + 1, 2*n + 1
    k = np.zeros((H, W), dtype=int)
    # make the center at (m,n) 
    k[:m, :n] = 1
    return k

from scipy.ndimage import correlate
kernel = kernel_left_up(leadD * 4 + 1, 11)  # (time_window, lon_window)
result = correlate(trackPoints_2D_resampled, kernel, mode='constant', cval=0)
trackPoints_2D_mask = (result > 0).astype(int)
print(trackPoints_2D_mask.shape, flush=True)
trackdays = np.nansum(trackPoints_2D_mask, axis=0) # number of days with tracks at each lon grid point
print('trackdays shape:', trackdays.shape, flush=True) # trackdays shape: (360,), a value each day
print('number of days with tracks at each lon:', trackdays)

# 05-3 get the blocking days with tracks preceding
# trackwithBlocking
trackBlk = blockingEidArr_6hr * trackPoints_2D_mask
print('arr0/1 blocking with track preceding shape:', trackBlk.shape, flush=True)
trackBlk_num = np.nansum(trackBlk,axis=0)
print('blocking with track preceding number at each lon:', trackBlk_num, flush=True) # (360,), a value each day
track_B_freq = (trackBlk_num/trackdays) # frequency of blocking under track days
print('frequency of blocking under track days out of all track days:', track_B_freq, flush=True)

#%% 06 get the probability under highBAM stage
highBAMindex = np.where(B_B_6hr[:,0]>0)[0]
print('highBAM location:', highBAMindex, flush=True)
trackPoints_2D_mask_HB = trackPoints_2D_mask[highBAMindex,:] # get the track points under high BAM condition
blockingEidArr_6hr_HB = blockingEidArr_6hr[highBAMindex,:]  # get the blocking events under high BAM condition
print('track points under high BAM condition 0/1 shape:', trackPoints_2D_mask_HB.shape, flush=True)
print('blocking events under high BAM condition 0/1 shape:', blockingEidArr_6hr_HB.shape, flush=True)
trackBlk_HB = blockingEidArr_6hr_HB * trackPoints_2D_mask_HB # get the track+blocking under high BAM condition
print('track+blocking under high BAM condition 0/1 shape:', trackBlk_HB.shape, flush=True)

trackBlk_HB_num = np.nansum(trackBlk_HB,axis=0)
trackdays_HB = np.nansum(trackPoints_2D_mask_HB, axis=0)
print('number of track+blocking under highBAM at each lon:', trackBlk_HB_num, flush=True)
print('number of track under highBAM at each lon:', trackdays_HB, flush=True)
track_B_freq_HB = (trackBlk_HB_num/trackdays_HB) # frequency of blocking under track days and high BAM condition

# make the final plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Blklon, B_B_freq, color='red', label=f'High BAM, n={int(np.round(np.nanmean(B_B_day/4)))} days')
ax.plot(Blklon, track_B_freq, color='blue', label=f'with ACs preceding, n={int(np.round(np.nanmean(trackdays/4)))} days')
ax.plot(Blklon, track_B_freq_HB, color='purple', label=f'ACs preceding + High BAM, n={int(np.round(np.nanmean(trackdays_HB/4)))} days')
ax.plot(Blklon, B_freq_clima, color='black', label=f'Climatology, n={int(np.round(ntime/4))} days')
# target region shading
# ax.axvspan(lon_min, lon_max, color='yellow', alpha=0.3, label=f'Study Region {lon_min}°-{lon_max}°')
# labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Blocking Frequency')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
plt.savefig(f'BAMwithACs_BlockingFreq_AlongLon_{rgname}_Type{typeid}_{cyc}_{leadD}daylead.png', dpi=300)

