import numpy as np
import datetime as dt
from datetime import date
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

# 01 read data
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc')
# ds = ds.sel(time=slice("1979-01-01", "1979-01-10"))
# ds.to_netcdf('/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979JFM_F128_subset.nc')

# z500 = ds['var129'].squeeze()/9.80665 # unit: m
z500 = ds['var129'].squeeze()

print(z500.shape)
# 02 calculate anomaly
zonal_mean = z500.mean(dim='lon') # zonal mean
zonal_mean_expanded = np.expand_dims(zonal_mean, axis=2) # expand the zonal mean 
Z_anomaly = z500-zonal_mean_expanded # z-zonal mean; (time,lat,lon)
print(Z_anomaly.shape,flush=True)
# get the monthly mean 
Z_anomaly_month = Z_anomaly.resample(time='1MS').mean() # monthly mean; (months,lat,lon) 
Z_anomaly_month_smoothed = Z_anomaly_month.rolling(time=3, center=True).mean() #(months,lat,lon)
print(Z_anomaly_month_smoothed.shape,flush=True)
# get the climatology
climatology = Z_anomaly_month_smoothed.groupby("time.month").mean(dim='time')
print(climatology.shape,flush=True)
ds2 = ds.copy()
ds2['var129'] = climatology
ds2.to_netcdf('/scratch/bell/hu1029/Data/processed/ERA5_geopotential500_climatology_monthly_1979_2021.nc')
print('climatology saved to netCDF file -----------------',flush=True)

# get the final Zanomaly
print(climatology['month'])
months = Z_anomaly['time'].dt.month 
print(months)
Zclimatology = climatology.sel(month=months) # expand the climatology to the same shape as the original data
Z_anomaly_corrected = Z_anomaly - Zclimatology

# 03 save data into netCDF file
# delete the original var129
ds = ds.drop_vars('var129')
ds['var129'] = Z_anomaly_corrected
ds = ds.drop_vars('month')

# write to the nc file
ds.to_netcdf('/scratch/bell/hu1029/Data/processed/ERA5_geopotential500_subtractseasonal_6hr_1979_2021.nc')

