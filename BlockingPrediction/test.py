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
import cv2
import copy
from collections import defaultdict

rgname = 'SP'

if rgname == "SP":
    with open(f'/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks_SH.pkl', 'rb') as file:
        AC_track_data = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks_SH.pkl', 'rb') as file:
        CC_track_data = pickle.load(file)
else:
    with open(f'/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks.pkl', 'rb') as file:
        AC_track_data = pickle.load(file)
    with open(f'/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks.pkl', 'rb') as file:
        CC_track_data = pickle.load(file)
    
print(AC_track_data[0][0])
print(AC_track_data[0][1])

print(AC_track_data[1])

print('done')
