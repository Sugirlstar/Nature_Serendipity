import numpy as np
import datetime as dt
from datetime import date
from datetime import datetime
from matplotlib import pyplot as plt
import cmocean
from HYJfunction import *
import math
import time

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
from scipy.ndimage import label
from scipy.interpolate import interp2d
import sys
import os

# %% function --------------------------------
regions = ["ATL", "NP", "SP"]
seasons = ["DJF", "JJA", "ALL"]
seasonsmonths = [[12, 1, 2], [6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
blkTypes = ["Ridge", "Trough", "Dipole"]
cycTypes = ["CC", "AC"]

ss = "ALL"
for eve in ['Blocking','Seeding']:
    for typeid in [1,2,3]:
        for rgname in regions:

            with open(f"/scratch/bell/hu1029/LGHW/SD_{eve}FlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}", "rb") as fp:
                ATLlist = pickle.load(fp)

            with open(f"{eve}_totalNumber.txt", "a") as f:
                f.write(f"Total {eve} number, Type{typeid}_{rgname}_{ss}: {len(ATLlist)}\n")
                print(f"Total {eve} number, Type{typeid}_{rgname}_{ss}: {len(ATLlist)}")
            
print('done')
