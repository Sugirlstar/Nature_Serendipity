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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# %% 00 prepare
regions = ["ATL", "NP", "SP"]
seasons = [ "ALL", "DJF", "JJA"]
seasonsmonths = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1, 2], [6, 7, 8]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["AC", "CC"]

# read in LWA
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # -90～90, it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

# get the first day and location of each blocking event, output as lists, three types:
for typeid in [1,2,3]:
    for rgname in regions:
        for ss in seasons:

            # attributes for z500
            lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
            lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
            lat_mid = int(len(lat)/2) + 1 
            if rgname == "SP":
                Blklat = lat[lat_mid:len(lat)]
                LWA_td = LWA_td_origin[:,lat_mid:len(lat),:] # keep only the SH
            else:
                Blklat = lat[0:lat_mid-1]
                LWA_td = LWA_td_origin[:,0:lat_mid-1,:] # keep only the NH
            Blklat = np.flip(Blklat) # make it ascending order (from south to north)
            LWA_td = np.flip(LWA_td, axis=1) # make it from north to south
            print(Blklat, flush=True)
            Blklon = lon 

            # attributes for Track
            ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
            timesarr = np.array(ds['time'])
            datetime_array = pd.to_datetime(timesarr)
            timei = list(datetime_array)
            print(len(timei))

            with open(f'/scratch/bell/hu1029/LGHW/SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}', "rb") as f:
                ATLlist = pickle.load(f)

            # read in all blocking events date list and label array
            if rgname == "SP":
                with open("/scratch/bell/hu1029/LGHW/SD_Blocking_diversity_date_daily_SH", "rb") as fp:
                    Blocking_diversity_date = pickle.load(fp)
                with open("/scratch/bell/hu1029/LGHW/SD_Blocking_diversity_label_daily_SH", "rb") as fp:
                    Blocking_diversity_label = pickle.load(fp)
            else:
                with open("/scratch/bell/hu1029/LGHW/SD_Blocking_diversity_date_daily_NH", "rb") as fp:
                    Blocking_diversity_date = pickle.load(fp)
                with open("/scratch/bell/hu1029/LGHW/SD_Blocking_diversity_label_daily_NH", "rb") as fp:
                    Blocking_diversity_label = pickle.load(fp)

            # event date and timeid list
            BlkeventList = Blocking_diversity_date[typeid-1]
            BlkeventList = [BlkeventList[x] for x in ATLlist] # get the target events only
            firstday_Date = [sublist[0] for sublist in BlkeventList] # a list of first day of each event
            first_day_id = [timei.index(date) for date in firstday_Date] # a list of the index of the first day of each event
            # note: the timei is a 6-hourly time list, while the firstday_Date is for daily time of blocking. 
            # However, the first day of blocking event is always at 00:00 UTC, so the index of the first day in timei can always be matched.
            # event label list
            BlkeventLabel = Blocking_diversity_label[typeid-1]
            BlkeventLabel = [BlkeventLabel[x] for x in ATLlist] # get the target events only
            firstdayLabel = [sublist[0] for sublist in BlkeventLabel] # a list of the label of the first day of each event

            lat_values = []
            lon_values = []

            for idx, i in enumerate(first_day_id):
                slice_2d = np.flip(np.array(firstdayLabel[idx]),axis=0) * LWA_td[i,:, :]
                max_idx = np.unravel_index(np.nanargmax(slice_2d), slice_2d.shape)  # （lat_idx, lon_idx）
                lat_values.append(Blklat[max_idx[0]])
                lon_values.append(Blklon[max_idx[1]])

            print(len(lon_values), flush=True)

            #%% Save results
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}", "wb") as fp:
                pickle.dump(firstday_Date, fp)
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLatList_blkType{typeid}_{rgname}_{ss}", "wb") as fp:
                pickle.dump(lat_values, fp)
            with open(f"/scratch/bell/hu1029/LGHW/Blocking1stdayLonList_blkType{typeid}_{rgname}_{ss}", "wb") as fp:
                pickle.dump(lon_values, fp)

print('done')
