#%%
###### This code is to reproduce Thompson's BAM result, based on MERRA2 dataset ######
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
#import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import glob
import copy
import pickle
import matplotlib.path as mpath
from netCDF4 import Dataset
from scipy.io import loadmat
from eofs.standard import Eof
from eofs.examples import example_data_path
from datetime import datetime, timedelta
from scipy.signal import butter,filtfilt, sosfilt

### Design a FFT_bandpass_filter ###
def bandpass(x,k,w):
    xf = np.fft.fft(x)
    xftmp=xf*0.0
    xftmp[k]=xf[k]
    xftmp[-k]=xf[-k]
    xout=np.fft.ifft(xftmp)
    return xout

### Time management ###
start_date = datetime(1979, 1, 1)
end_date = datetime(2022, 1, 1)  # 20211231
print(start_date)

nday = (end_date - start_date).days

#%%
###### The next step is to calculate daily EKE ######
### First we get the directory of each nc data ###
#files = glob.glob(r"/home/zhao1550/scratch/ERA5_EKE_synoptic/*.nc")
files = glob.glob(r"/scratch/bell/hu1029/Data/processed/ERA5_EKE_total_1979_2021/SH*.nc")
files.sort()
N = len(files)   #The number of u nc files

### read any data file to read some basic variables like lon and lat ###
fil = Dataset(files[0],'r')
lon = fil.variables['lon'][:]
lat = fil.variables['lat'][:]
lev0 = fil.variables['level'][:]
fil.close()

ilat = np.where((lat >= -70) & (lat <= -20))[0]
ilev = np.where((lev0 >= 200) & (lev0 <= 1000))[0]
print(ilat)
print(ilev)

lev = lev0[ilev]     ### 1000-200hPa
lat = lat[ilat]   ### 20N to 70N
nlev = len(lev)
nlat = len(lat)
nlon = len(lon)
#%%
### Read the zonal-mean EKE (20-70N, 1000hPa-200hPa) ###
EKE = np.zeros((nday,nlev,nlat))

# Concatenate the data
day_offset = 0  # Keep track of the starting index for each file
for year, file in enumerate(files):
    with Dataset(file, 'r') as fil:
        # Get the number of days in the current file
        n_days_in_file = len(fil.variables['time'])
        
        # Extract EKE and compute zonal mean
        EKE_zonal_mean = fil.variables['EKE'][:,ilev[:],ilat[:],:].mean(axis=3)
        
        # Assign to the main EKE array
        EKE[day_offset:day_offset + n_days_in_file, :, :] = EKE_zonal_mean
        day_offset += n_days_in_file  # Update the day offset
    
    print(f"Processed file {year + 1}/{N}: {file}", flush=True)
    
np.save('/scratch/bell/hu1029/Data/processed/ERA5_EKE_ZM_SH_total_1979_2021_1dg.npy',EKE)



