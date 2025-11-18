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
from scipy.stats import pearsonr

# %% 00 prepare the environment and functions
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA","ALL"]
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

def extendList(desired_length,centorinlist,test):
    targetlist = np.full(desired_length*2+1, np.nan)
    targetlist[desired_length] = test[centorinlist]
    leftlen = centorinlist
    rightlen = len(test) - leftlen -1
    left_padding = [np.nan] * (desired_length - leftlen)
    left_values = left_padding + list(test[np.nanmax([0, leftlen-desired_length]):centorinlist])
    right_padding = [np.nan] * (desired_length - rightlen)
    if rightlen-desired_length>0:
        rightend = centorinlist+1+desired_length
    else: rightend = len(test)
    right_values = list(test[centorinlist+1:rightend]) + right_padding
    targetlist[:desired_length] = left_values
    targetlist[desired_length+1:] = right_values
    return targetlist

ss = 'ALL'
for typeid in [1,2,3]:
     for cyc in cycTypes:
        for rgname in regions:

            print(f'start: Blocking type{typeid} - {cyc} - {rgname} - {ss}')
            intopercentList = np.load(f'/scratch/bell/hu1029/LGHW/intopercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy')
            interactingduration = np.load(f'/scratch/bell/hu1029/LGHW/interactingduration_type{typeid}_{cyc}_{rgname}_{ss}.npy')
            print(f'{typeid}_{cyc}_{rgname}_{ss}, intopercent<=20% percentage: {len(np.where(intopercentList<=0.2)[0])/len(intopercentList)}', flush=True)
            print(f'{typeid}_{cyc}_{rgname}_{ss}, duration<=2 percentage: {len(np.where(interactingduration<=2)[0])/len(interactingduration)}', flush=True)

            with open("EnterStay_Statistic.txt", "a") as f:
                f.write(f'{typeid}_{cyc}_{rgname}_{ss}, intopercent<=20% percentage: {len(np.where(intopercentList<=0.2)[0])/len(intopercentList)}\n')
                f.write(f'{typeid}_{cyc}_{rgname}_{ss}, duration<=2 percentage: {len(np.where(interactingduration<=2)[0])/len(interactingduration)}\n')

            # test2: KDE+scatter plot of entry time and blocking duration+blocking duration pdf
                



                