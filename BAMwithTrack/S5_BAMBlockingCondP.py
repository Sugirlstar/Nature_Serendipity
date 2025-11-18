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

#%% 01 read the data
# time management
Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
nday = len(Date0)
Month = Date0['date'].dt.month
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
Datelist = list(Date0)
# BAM dates
feb_29_ind = Date0[(Month == 2) & (Day == 29)].index
# Remove February 29 from the Date DataFrame
print(Date0.shape, flush=True)
BAMDate = Date0.drop(index=feb_29_ind).reset_index(drop=True)
print(BAMDate.shape, flush=True)

# read the BAM index
BI = np.load('/scratch/bell/hu1029/LGHW/SBAM_index_total_no_leap.npy')                             ### This is full BAM index from 1979 to 2021, 15695 in total, no leap years ###
print(BI.shape, flush=True)

# read the SH Blocking index 
B_freq_2d = np.load('/scratch/bell/hu1029/LGHW/B_freq_SH.npy')
B_freq_3d = np.load('/scratch/bell/hu1029/LGHW/B_freq2_SH.npy')

#%% identify BAM phases
BI_copy = copy.copy(BI)
BAM_event_all = []      ## To store the date or index with large pc1 value ##
BAM_event_BI_all = []   ## The value of the BI at that peaking date ##
number = 0
while 1:
    index = np.squeeze(np.array(np.where( BI_copy==np.nanmax(BI_copy) )))[0]
    if BI_copy[index]>=BI[index+1] and BI[index]>=BI[index-1] and BI_copy[index]>=BI[index+2] and BI[index]>=BI[index-2]:        
        BAM_event_all.append(index)
        BAM_event_BI_all.append(BI[index])
        number+=1
        BI_copy[index-12:index+13,0] = np.nan
        if number > 550:
            break
    else:
        BI_copy[index]=np.nan
        
print(BAM_event_all, flush=True)
print(BAM_event_BI_all, flush=True)
print(len(BAM_event_all), flush=True)
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
print(BAM_event_all, flush=True)
print(BAM_event_BI_all, flush=True)
print(len(BAM_event_all), flush=True)

T=1
n_BAM = len(BAM_event_all)
# B_B and B_LB is to store the blocking condition under BAM or low BAM condition days
B_B = np.zeros((BI.shape[0],B_freq_3d.shape[1],B_freq_3d.shape[2]))  ## This array is to store the blocking which falls into +-T day of BAM peaking date
B_LB = np.zeros((BI.shape[0],B_freq_3d.shape[1],B_freq_3d.shape[2])) ## This array is to store the blocking whihc falls into +-T day of low BAM date
for i in np.arange(n_BAM):
    # the date and index for BAM and Blocking are different, cause of the 0229. So we need to match the exact dates
    BAMloc = BAM_event_all[i]
    print('BAMloc:' ,BAMloc, flush=True)
    BAMday = BAMDate.iloc[BAMloc]
    print('BAMday:' ,BAMday, flush=True)
    BfreqLoc = np.where((Date0['date'] == BAMday['date']))[0][0]  ## This is the index of the BAM date in the Date0 DataFrame
    print('BfreqLoc:' ,BfreqLoc, flush=True)
    B_B[BAMloc-T:BAMloc+T+1,:,:] = B_freq_3d[BfreqLoc-T:BfreqLoc+T+1,:,:]
    B_LB[BAMloc-12-T:BAMloc-12+T+1,:,:] = B_freq_3d[BfreqLoc-12-T:BfreqLoc-12+T+1,:,:]

#%% make the plot
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #91
lat_SH = lat[lat_mid:len(lat)]
print(lat_SH.shape, flush=True)
lat_mask = (lat_SH >= -70) & (lat_SH <= -20)
lat_target = lat_SH[lat_mask]
print(B_freq_2d.shape, flush=True)

B_B_day = n_BAM*(2*T+1)
B_B_num = B_B.sum(axis=0)
B_LB_num = B_LB.sum(axis=0)
B_B_freq = (B_B_num/B_B_day)
B_LB_freq = (B_LB_num/B_B_day)
B_freq_clima = B_freq_2d/nday
            
minlev =0
maxlev = B_B_freq[lat_mask,:].max()
levs = np.linspace(0.02,0.16,20)

proj=ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=[10,7])

ax = fig.add_subplot(3,1,1, projection=proj)
h1 = plt.contourf(lon, lat_target, B_freq_clima[lat_mask,:], levs, transform=ccrs.PlateCarree(), cmap='hot_r')  #_r is reverse
# plt.xlabel('longitude',fontsize=10)
plt.ylabel('latitude',fontsize=10)
plt.title("(a) Blocking Frequency (Climatology)", pad=5)
ax.coastlines()
ax.gridlines(linestyle="--", alpha=0.7)
ax.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
ax.set_yticks([-80,-60,-40,-20], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter) 
bx= fig.add_subplot(3,1,2, projection=proj)
h1 = plt.contourf(lon, lat_target, B_B_freq[lat_mask,:], levs, transform=ccrs.PlateCarree(), cmap='hot_r')  #_r is reverse
bx.coastlines()
bx.gridlines(linestyle="--", alpha=0.7)
bx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
bx.set_yticks([-80,-60,-40,-20], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter) 
# plt.xlabel('longitude',fontsize=10)
plt.ylabel('latitude',fontsize=10)
plt.title("(b) Conditional Blocking Frequency (High BAM State)", pad=5)

cx = fig.add_subplot(3,1,3, projection=proj)
h2 = plt.contourf(lon, lat_target, B_LB_freq[lat_mask,:], levs, transform=ccrs.PlateCarree(), cmap='hot_r')  #_r is reverse
plt.xlabel('longitude',fontsize=10)
plt.ylabel('latitude',fontsize=10)
plt.title("(c) Conditional Blocking Frequency (Low BAM State) ", pad=5)
cx.coastlines()
cx.gridlines(linestyle="--", alpha=0.7)
cx.set_xticks([0,60,120,180,240,300,358.5], crs=ccrs.PlateCarree())
cx.set_yticks([-80,-60,-40,-20], crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label='FALSE')
lat_formatter = LatitudeFormatter()
cx.xaxis.set_major_formatter(lon_formatter)
cx.yaxis.set_major_formatter(lat_formatter) 

plt.subplots_adjust(hspace=0.5, right=0.8)

cbar = fig.add_axes([0.85,0.1,0.015,0.8])
cb = plt.colorbar(h1, cax=cbar, ticks=[0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16])
cb.set_label('frequency',fontsize=10)

plt.savefig("BlockingPeakingPredict_Fig3.png",dpi=600)
plt.close()


print('all done')
