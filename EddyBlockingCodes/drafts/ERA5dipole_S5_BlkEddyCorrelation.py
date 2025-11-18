import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math

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
import imageio
from scipy import stats
from collections import defaultdict
from scipy.ndimage import binary_dilation, generate_binary_structure, iterate_structure

# check two correlations: 
# 1. Blocking's persistence ~ the number of the related track points
# 2. Blocking's LWA ~ the eddy's LWA

def findClosest(lati, latids):
    if isinstance(lati, np.ndarray):  # if lat is an array
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

def radiusExpand(arr, radius):
    
    structure = generate_binary_structure(2, 1)
    structure = iterate_structure(structure, radius)
    # wrap-around dilation
    expanded_arr = np.zeros_like(arr)
    for t in range(arr.shape[0]):
        slice2d = arr[t]
        padded = np.concatenate([slice2d[:, -radius:], slice2d, slice2d[:, :radius]], axis=1)
        # binary dilation on padded
        dilated = binary_dilation(padded, structure=structure).astype(int)
        expanded_arr[t] = dilated[:, radius:-radius]
    return expanded_arr

# 01 load the data -------------------------------------------------------------
# read in LWA
LWA_td = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td = LWA_td/100000000 # change the unit to 1e8 
LWA_td = LWA_td[:,0:90,:] # keep only the NH
LWA_td = np.flip(LWA_td, axis=1) # make it from north to south
print('-------- LWA loaded --------', flush=True)
print(LWA_td.shape, flush=True)

# # Time management
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
timesarr = np.array(ds['time'])
datetime_array = pd.to_datetime(timesarr)
timei = list(datetime_array)
print(len(timei))
# lat and lon for track points
lon = np.array(ds['lon'])
lat = np.array(ds['lat'])
lat = np.flip(lat)
latNH = lat[(findClosest(0,lat)+1):len(lat)] # print(len(latNH))

# read in all blocking events date list and label array - Daily!
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_date_daily", "rb") as fp:
    Blocking_diversity_date = pickle.load(fp)      
with open("/scratch/bell/hu1029/LGHW/Blocking_diversity_label_daily", "rb") as fp:
    Blocking_diversity_label = pickle.load(fp)   
# read in all ATL blocking event id list
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_1_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist1 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_2_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist2 = pickle.load(fp)
with open(f"/scratch/bell/hu1029/LGHW/ERA5dipoleDaily_BlockingType_3_ATLeventList_newATLdefine", "rb") as fp:
    ATLlist3 = pickle.load(fp)
all_lists = [ATLlist1, ATLlist2, ATLlist3]

# lat and lon for blockings and LWA
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
Blklon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 #91
Blklat = lat[0:lat_mid-1]
Blklat = np.flip(Blklat)

# for each blocking event, get the persistence and the number of related track points, and LWA
for typeid in [1,2,3]:

    # get the interacting Eddy track points - only interacting tracks remained - values: the track's ID
    CCtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/CCtrackPoints_TrackIDarray_inBlkType{typeid}.npy')
    ACtrackpoints = np.load(f'/scratch/bell/hu1029/LGHW/ACtrackPoints_TrackIDarray_inBlkType{typeid}.npy')
    print('all trackpoints loaded', flush=True)

    # regrid the track point to the blocking grid
    # CC
    CCarrFlag = np.zeros_like(LWA_td)  # create an array to store the track points in the blocked area
    Locti,lati,loni = np.where(CCtrackpoints>0)
    latinLWA = findClosest(latNH[np.array(lati,dtype=int)], Blklat)
    loninLWA = findClosest(lon[np.array(loni,dtype=int)], Blklon)
    CCarrFlag[Locti, latinLWA, loninLWA] = CCtrackpoints[Locti,lati,loni]  # get the track points in the blokcing grid
    CCtrackpoints = CCarrFlag
    # AC
    ACarrFlag = np.zeros_like(LWA_td)  # create an array to store the track points in the blocked area
    Locti,lati,loni = np.where(ACtrackpoints>0)
    latinLWA = findClosest(latNH[np.array(lati,dtype=int)], Blklat)
    loninLWA = findClosest(lon[np.array(loni,dtype=int)], Blklon)
    ACarrFlag[Locti, latinLWA, loninLWA] = ACtrackpoints[Locti,lati,loni]  # get the track points in the blokcing grid
    ACtrackpoints = ACarrFlag

    type_idx = typeid - 1
    ATLlist = all_lists[type_idx]  # get the ATL list for the current typeid
    # event date and timeid list
    BlkeventList = Blocking_diversity_date[type_idx]
    BlkeventList = [BlkeventList[x] for x in ATLlist] # get the target events time
    # event label list
    BlkeventLabel = Blocking_diversity_label[type_idx]
    BlkeventLabel = [BlkeventLabel[x] for x in ATLlist] # get the target events label

    # statistics for blocking events
    BlkPersistence = [len(sublist) for sublist in BlkeventList] # Blocking event's persistence
    
    BlkeventLWA = [] # Blocking event's LWA
    BlkeventCCNums = [] # CC's number during the events
    BlkeventACNums = [] # AC's number during the events
    BlkeventCCLWA = [] # CC's LWA during the events
    BlkeventACLWA = [] # AC's LWA during the events
    BlkACbeforeNum = [] # AC's before the blocking event
    BlkCCbeforeNum = [] # CC's before the blocking event
    BlkACbeforeLWA = [] # AC's before the blocking event LWA
    BlkCCbeforeLWA = [] # CC's before the blocking event LWA
    for i, event in enumerate(BlkeventList):

        persis = BlkPersistence[i]  # the persistence of the current blocking event
        print('checkpoint1-persis:', persis, flush=True)
        # get the time indices
        timeindex1 = timei.index(event[0])  # the first timepoint of the blocking event
        timeindex2 = timei.index(event[-1]) + 3  # the last timepoint of the blocking event, +3 for get the last 6-hourly
        times = list(range(timeindex1, timeindex2 + 1))  # the time range of the blocking event, for 6-hourly
        # get the event's LWA (sum over the time range)
        eventLabel = np.array(BlkeventLabel[i])
        eventLabel = np.repeat(eventLabel, repeats=4, axis=0) # repeat the label to be 6-hourly
        eventLabel = np.flip(eventLabel, axis = 1) 
        BlkLWA =  eventLabel * LWA_td[times, :, :]
        BlkdailyLWA = np.nansum(BlkLWA, axis=(1,2))  # sum the LWA over the time range
        eventLWA = np.nanmean(BlkdailyLWA)  # average the LWA everyday
        BlkeventLWA.append(eventLWA)
        print('checkpoint3-eventLWA:', eventLWA, flush=True)

        # get the track point nums 
        CCarrFlag = CCtrackpoints[times,:,:]  # get the track points in the blokcing grid
        BlkCCarr = eventLabel * CCarrFlag
        trackuqid = np.unique(BlkCCarr[BlkCCarr > 0])  # get the unique track IDs
        BlkeventCCNums.append(len(trackuqid))
        print('checkpoint4-CCNums:', len(trackuqid), flush=True)

        ACarrFlag = ACtrackpoints[times,:,:]
        BlkACarr = eventLabel * ACarrFlag  # get the track points in the blocked area
        trackuqid = np.unique(BlkACarr[BlkACarr > 0])  # get the unique track IDs
        BlkeventACNums.append(len(trackuqid))
        print('checkpoint5-ACNums:', len(trackuqid), flush=True)

        # get all the track points' LWA
        BlkCCarr[np.where(BlkCCarr > 0)] = 1  # set the track points to 1
        print('checkpoint6-CCtrackpointgridnum_origin:', np.sum(BlkCCarr > 0))
        BlkCCarrX = radiusExpand(BlkCCarr, radius=5)  # expand the trackpoints to the surrounding area
        print('checkpoint7-CCtrackpointgridnum_expanded:', np.sum(BlkCCarrX > 0))
        BlkCCLWAarr = BlkCCarrX * LWA_td[times,:,:]  # get the LWA of the track points
        dailyLWA = np.nansum(BlkCCLWAarr, axis=(1,2))  # sum the LWA over the time range
        CCLWA = np.nanmean(dailyLWA)  # sum the LWA of the track points over the time range
        BlkeventCCLWA.append(CCLWA)
        print('checkpoint8-CCLWA:', CCLWA, flush=True)

        BlkACarr[np.where(BlkACarr > 0)] = 1  # set the track points to 1
        print('checkpoint9-ACtrackpointgridnum_origin:', np.sum(BlkACarr > 0))
        BlkACarrX = radiusExpand(BlkACarr, radius=5)  # expand the trackpoints to the surrounding area
        print('checkpoint10-ACtrackpointgridnum_expanded:', np.sum(BlkACarrX > 0))
        BlkACLWAarr = BlkACarrX * LWA_td[times,:,:]  # get the LWA of the track points
        dailyLWA = np.nansum(BlkACLWAarr, axis=(1,2))  # sum the LWA over the time range
        ACLWA = np.nanmean(dailyLWA)  # sum the LWA of the track points over the time range
        BlkeventACLWA.append(ACLWA)
        print('checkpoint11-ACLWA:', ACLWA, flush=True)

        # get the 1st day's eddies' tracks before that day
        AnchorDay = 0
        secondindex = timeindex1 + AnchorDay*4

        BlkAnchorACs = ACtrackpoints[secondindex,:,:] * eventLabel[AnchorDay,:,:]
        # find the unique track IDs in the second day's eddies
        unique_ACs = np.unique(BlkAnchorACs[BlkAnchorACs > 0])  # get the unique track IDs
        print('checkpoint12-unique_ACs_beforeEnter:', unique_ACs, flush=True)
        beforenum = len(unique_ACs)  # the number of unique track IDs
        BlkACbeforeNum.append(beforenum)
        # find the earierist point of these tracks
        tp = []
        beforeLWAmean = np.nan
        if len(unique_ACs) > 0:  # if there are unique track IDs
            for trackid in unique_ACs:
                ti, _, _ = np.where(ACtrackpoints == trackid)
                tp.append(ti[0])
            mintimepoint = np.nanmin(tp)  # the earliest time point of the tracks
            BlkACbeforeArr = ACtrackpoints[mintimepoint:secondindex,:,:]
            BlkACbeforeArr[np.where(BlkACbeforeArr>0)] = 1  # set the track points to 1
            BlkACbeforeArrX = radiusExpand(BlkACbeforeArr, radius=5)  # expand the trackpoints to the surrounding area
            beforeLWA = BlkACbeforeArrX * LWA_td[mintimepoint:secondindex,:,:]  # get the LWA of the track points
            beforeLWAdaily = np.nansum(beforeLWA, axis=(1,2))
            beforeLWAmean = np.nanmean(beforeLWAdaily)
        BlkACbeforeLWA.append(beforeLWAmean)  # sum the LWA of the track points over the time range
        print('checkpoint13-ACbeforeLWA:', beforeLWAmean, flush=True)

        BlkAnchorCCs = CCtrackpoints[secondindex,:,:] * eventLabel[AnchorDay,:,:]
        # find the unique track IDs in the second day's eddies
        unique_CCs = np.unique(BlkAnchorCCs[BlkAnchorCCs > 0])  # get the unique track IDs
        print('checkpoint14-unique_CCs_beforeEnter:', unique_CCs, flush=True)
        beforenum = len(unique_CCs)  # the number of unique track IDs
        BlkCCbeforeNum.append(beforenum)
        # find the earierist point of these tracks
        tp = []
        beforeLWAmean = np.nan
        if len(unique_CCs) > 0:  # if there are unique track IDs
            for trackid in unique_CCs:
                ti, _, _ = np.where(CCtrackpoints == trackid)
                tp.append(ti[0])
            mintimepoint = np.nanmin(tp)
            BlkCCbeforeArr = CCtrackpoints[mintimepoint:secondindex,:,:]
            BlkCCbeforeArr[np.where(BlkCCbeforeArr>0)] = 1
            BlkCCbeforeArrX = radiusExpand(BlkCCbeforeArr, radius=5)
            beforeLWA = BlkCCbeforeArrX * LWA_td[mintimepoint:secondindex,:,:]
            beforeLWAdaily = np.nansum(beforeLWA, axis=(1,2))
            beforeLWAmean = np.nanmean(beforeLWAdaily)
        BlkCCbeforeLWA.append(beforeLWAmean)  # sum the LWA of the track points over the time range
        print('checkpoint15-CCbeforeLWA:', beforeLWAmean, flush=True)

        print('-------')

    #%% Save results
    variables = {
        'BlkPersistence': BlkPersistence,
        'BlkeventLWA': BlkeventLWA,
        'BlkeventCCNums': BlkeventCCNums,
        'BlkeventACNums': BlkeventACNums,
        'BlkeventCCLWA': BlkeventCCLWA,
        'BlkeventACLWA': BlkeventACLWA,
        'BlkACbeforeNum': BlkACbeforeNum,
        'BlkCCbeforeNum': BlkCCbeforeNum,
        'BlkACbeforeLWA': BlkACbeforeLWA,
        'BlkCCbeforeLWA': BlkCCbeforeLWA,
    }

    save_dir = '/scratch/bell/hu1029/LGHW/'
    for name, var in variables.items():
        filename = f"{save_dir}ERA5dipoleDaily_typeid{typeid}_{name}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(var, f)

    #%% Correlation analysis
    df = pd.DataFrame(variables)
    df = df.fillna(0)
    df_zscore = (df - df.mean()) / df.std()
    n = df_zscore.shape[1]
    corr_matrix = np.zeros((n, n))
    pval_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r, p = stats.pearsonr(df_zscore.iloc[:, i], df_zscore.iloc[:, j])
            corr_matrix[i, j] = r
            pval_matrix[i, j] = p
    print(corr_matrix, flush=True)

    corr_df = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
    pval_df = pd.DataFrame(pval_matrix, index=df.columns, columns=df.columns)

    def significance_mask(pvals, threshold=0.05):
        return np.where(pvals < threshold, '*', '')

    stars = significance_mask(pval_matrix)

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    ax = sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    cbar_kws={"label": "Pearson r"}, annot_kws={"size": 10}, linewidths=0.5, linecolor='gray')
    # # significant stars
    # for i in range(n):
    #     for j in range(n):
    #         if i != j and stars[i, j] == '*':
    #             ax.text(j + 0.5, i + 0.5, '*', color='black', ha='center', va='center', fontsize=16, weight='bold')

    plt.title('Correlation Matrix (NaNs→0, Standardized)')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'ERA5dipole_typeid{typeid}_correlation_matrix.png', dpi=300)

    
    print(f"Saved: typeid{typeid}")





