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
Month = Date0['date'].dt.month
Year = Date0['date'].dt.year
Day = Date0['date'].dt.day
# BAM dates
feb_29_ind = Date0[(Month == 2) & (Day == 29)].index
print(feb_29_ind, flush=True)

for k in ['SH','NH']:
    
    print(f'Processing {k} ...', flush=True)

    # read the BAM index
    BI = np.load(f'/scratch/bell/hu1029/LGHW/{k}_BAM_index_total_no_leap.npy')                             ### This is full BAM index from 1979 to 2021, 15695 in total, no leap years ###
    print('BI shape before (No 0229):', len(BI), flush=True)
    print(BI.shape, flush=True)

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

    # identify BAM phases
    BI_copy = copy.copy(BI)
    BAM_event_all = []      ## To store the date or index with large pc1 value ##
    BAM_event_BI_all = []   ## The value of the BI at that peaking date ##
    number = 0
    while 1:
        if np.all(np.isnan(BI_copy)):
            break
        index = np.array(np.where( BI_copy==np.nanmax(BI_copy) ))[0][0] # start from the largest value
        if index < 12 or index > len(BI)-13: # avoid the edge 
            BI_copy[index] = np.nan
            continue
        if BI_copy[index]>=BI[index+1] and BI_copy[index]>=BI[index-1] and BI_copy[index]>=BI[index+2] and BI_copy[index]>=BI[index-2]:        
            BAM_event_all.append(index)
            BAM_event_BI_all.append(BI[index])
            number+=1
            BI_copy[index-12:index+13] = np.nan
            if number > 550: # find at most 550 events
                break
        else:
            BI_copy[index]=np.nan
            
    print(BAM_event_all[0:10], flush=True)
    print(BAM_event_BI_all[0:10], flush=True)
    print(f' {k} peaking length initially found: ',len(BAM_event_all), flush=True)
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
    print(BAM_event_all[0:10], flush=True)
    print(BAM_event_BI_all[0:10], flush=True)
    print(f' {k} peaking length after removing the repeat values: ',len(BAM_event_all), flush=True)

    # get the low BAM events: 12 days befor the peak days
    BAM_event_low_all = []
    BAM_event_low_BI_all = []
    for idx in BAM_event_all:
        BAM_event_low_all.append(idx-12)
        BAM_event_low_BI_all.append(BI[idx-12])
    
    # save the BAM event peak and low day as list
    with open(f'/scratch/bell/hu1029/LGHW/{k}_BAM_event_peak_list.pkl', 'wb') as f:
        pickle.dump(BAM_event_all, f)
    with open(f'/scratch/bell/hu1029/LGHW/{k}_BAM_event_low_list.pkl', 'wb') as f:
        pickle.dump(BAM_event_low_all, f)
    # save the BAM index with full time (with leap years)
    np.save(f'/scratch/bell/hu1029/LGHW/{k}_BAM_index_total_with_leap.npy', BI)   ### This is full BAM index from 1979 to 2021, 15695 in total, with leap years ###
    print(f'Saved {k} BAM index with leap years and event lists.', flush=True)
    