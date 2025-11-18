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


#%% 01 read the data ---------------------------------------------------------
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
print(feb_29_ind, flush=True)
# Remove February 29 from the Date DataFrame
print(Date0.shape, flush=True)
# BAMDate = Date0.drop(index=feb_29_ind).reset_index(drop=True) # BAMDate: the Date index for BAM index, without February 29
# print(BAMDate.shape, flush=True)

# read the BAM index
BI = np.load('/scratch/bell/hu1029/LGHW/SBAM_index_total_no_leap.npy')    ## This is full BAM index from 1979 to 2021, no leap years ###
print(BI.shape, flush=True)
print(BI[423],BI[424],BI[425],flush=True)  

# get back to the full time
BI_full = np.full(len(Date0), np.nan)
keep_idx = np.setdiff1d(np.arange(len(Date0)), feb_29_ind)
BI_full[keep_idx] = BI.squeeze()
BI_full_series = pd.Series(BI_full)
BI_full_interp = BI_full_series.interpolate(method='linear')
BI_full_interp = BI_full_interp.to_numpy()
print(BI_full_interp.shape)
print(BI_full_interp[423],BI_full_interp[424],BI_full_interp[425],flush=True)  
BI = BI_full_interp
print('***check*** BAM index shape: ', BI.shape, flush=True)

# make a plot to see
plt.figure(figsize=(12, 3))
plt.plot(np.arange(360), BI[0:360], label='BAM index')
plt.ylim(-3, 3)
plt.xlabel('Date')
plt.ylabel('BAM index')
plt.title('BAM index Time Series 360 days')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('BI_check.png', dpi=300, bbox_inches='tight')
plt.close()


#%% 02 read the blocking data (blocking grid flag) --------------------------------
# transfer to 2d array (time, lon)
typeid = 1
rgname = "SP"
ss = "ALL"
cyc = "AC"

# read in the blocking flag, and transfer to 2d array (time, lon)
blockingEidArr = np.load(f'/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{typeid}_{rgname}_{ss}.npy') # the blocking event index array
blockingEidArr = np.flip(blockingEidArr, axis=1) # flip the lat
print(blockingEidArr.shape, flush=True)
blockingEidArr = np.where(blockingEidArr != -1, 1, 0) # transfer to 0/1
blockingEidArr_2D = np.any(blockingEidArr, axis=1).astype(int)

lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
    Blklat = lat[lat_mid:len(lat)]
else:
    Blklat = lat[0:lat_mid-1]
Blklat = np.flip(Blklat) # make it ascending order (from south to north)
print('Blklat: ',Blklat, flush=True)
Blklon = lon 
print('Blklon: ',Blklon, flush=True)
print('***check*** blockingEidArr_2D shape: ', blockingEidArr_2D.shape, flush=True)

# read in LWA data
LWA_td_origin = np.load('/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy')  # [0]~[-1] it's from south to north
LWA_td_origin = LWA_td_origin/100000000 # change the unit to 1e8 

T = LWA_td_origin.shape[0]
n = T // 4
arr_reshaped = LWA_td_origin.reshape(n, 4, *LWA_td_origin.shape[1:])
LWA_td_origin_daily = np.nanmean(arr_reshaped, axis=1)  

# read in lat and lon for LWA
# get the lat and lon for LWA, grouped by NH and SH
lat_mid = int(len(lat)/2) + 1 
if rgname == "SP":
    latLWA = lat[lat_mid:len(lat)]
    LWA_td = LWA_td_origin_daily[:,lat_mid:len(lat),:] 
else:
    latLWA = lat[0:lat_mid-1]
    LWA_td = LWA_td_origin_daily[:,0:lat_mid-1,:]
latLWA = np.flip(latLWA) # make it ascending order (from south to north)
LWA_td = np.flip(LWA_td, axis=1) 
print('LWA shape: ', LWA_td.shape, flush=True)
lonLWA = lon 
print('latLWA:', latLWA, flush=True)
print('lonLWA:', lonLWA, flush=True)
ilat = np.where((latLWA >= -70) & (latLWA <= -20))[0]

# transfer the LWA to 2d array (time, lon)
LWA_td_2D = np.nanmean(LWA_td[:, ilat, :], axis=1)
print('***check*** LWA_td_2D shape: ', LWA_td_2D.shape, flush=True)

# get the LWA_2d anomaly as the deviation from the climalogical mean
LWA_td_clim = np.full((12, LWA_td_2D.shape[1]), np.nan)
for m in range(12):
    month_idx = np.where(Month == m+1)[0]
    LWA_td_clim[m,:] = np.nanmean(LWA_td_2D[month_idx,:], axis=0)
LWA_td_anom_2d = np.full(LWA_td_2D.shape, np.nan)
LWA_dailyclim = np.full(LWA_td_2D.shape, np.nan)
for i in range(LWA_td_2D.shape[0]):
    m = Month[i]
    LWA_td_anom_2d[i,:] = LWA_td_2D[i,:] - LWA_td_clim[m-1,:]
    LWA_dailyclim[i,:] = LWA_td_clim[m-1,:]
print('***check*** LWA_td_anom_2D shape: ', LWA_td_anom_2d.shape, flush=True)

# get the key longitude index
targetlon = 300
keylon = np.where(lonLWA == targetlon)[0][0]
print('keylon: ', keylon, flush=True)
keyLWA = LWA_td_anom_2d[:, keylon]
print('***check*** keyLWA shape: ', keyLWA.shape, flush=True)
# make a plot to see
plt.figure(figsize=(12, 3))
plt.plot(np.arange(360), keyLWA[0:360], label='BAM index')
plt.ylim(-3, 3)
plt.xlabel('Date')
plt.ylabel('LWA at 250')
plt.title('LWA Time Series 360 days at 250E')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('LWA_check.png', dpi=300, bbox_inches='tight')
plt.close()

keyBlkflag = blockingEidArr_2D[:, keylon]
# get the Xth LWA value during blocking days
thX = 50
keyLWA_blk_index = np.where(keyBlkflag == 1)[0]
print('keyLWA_blk_index: ', keyLWA_blk_index, flush=True)
keyLWAorigin = LWA_td_2D[:, keylon]
LWAvalues = keyLWAorigin[keyLWA_blk_index]
print('LWAvalues in blocking day: ', LWAvalues, flush=True)
# get the Xth percentile value
blkLWA = np.nanpercentile(LWAvalues, thX)
print('threshold blkLWA (10th): ', blkLWA, flush=True)

# make a plot to see
blkif = np.where(keyBlkflag[0:1000] == 1)[0]
plt.figure(figsize=(12, 3))
plt.plot(np.arange(1000), keyLWAorigin[0:1000], label='BAM index')
plt.plot(np.arange(1000)[blkif], keyLWAorigin[0:1000][blkif], 'ro', label='blocking days')
plt.ylim(-3, 3)
plt.xlabel('Date')
plt.ylabel('LWA at 250')
plt.title('LWA Time Series 1000 days at 250E')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig('LWAorigin_check.png', dpi=300, bbox_inches='tight')
plt.close()

# %%03 build the model using keyLWA and BI ---------------------------
# --- inputs ---
# keyLWA: 1D numpy array (daily LWA anomalies, already anomaly)
# BI    : 1D numpy array (daily BAM index)

y_full = np.asarray(keyLWA, float)
x_full = np.asarray(BI, float)

# drop NaNs
mask = np.isfinite(y_full) & np.isfinite(x_full)
y_full = y_full[mask]
x_full = x_full[mask]

# standardize BAM
x_full = (x_full - x_full.mean()) / (x_full.std() + 1e-12)

# ======================================================
# 1) Red noise model (AR(1) only)
# ======================================================
y_t   = y_full[1:]
y_tm1 = y_full[:-1]

# regression without intercept
phi_rn = np.dot(y_tm1, y_t) / np.dot(y_tm1, y_tm1)
resid_rn = y_t - phi_rn * y_tm1
sigma_rn = resid_rn.std()

print("=== Red Noise Model (AR(1), no intercept) ===")
print(f"phi (AR1):     {phi_rn:.4f}")
print(f"sigma (noise): {sigma_rn:.4f}")

# --- simulate red-noise series ---
N = len(y_full)
y_sim_rn = np.zeros(N)
rng = np.random.default_rng(42)
eps = rng.normal(0.0, sigma_rn, size=N)

# initialize with observed first value
y_sim_rn[0] = y_full[0]
for t in range(1, N):
    y_sim_rn[t] = phi_rn * y_sim_rn[t-1] + eps[t]

# ======================================================
# 2) Red noise + BAM forcing model (ARX(1))
# ======================================================
x_t   = x_full[1:]

X_arx = np.column_stack([y_tm1, x_t, np.ones_like(y_t)])
theta_arx = np.linalg.lstsq(X_arx, y_t, rcond=None)[0]
phi_arx, beta_arx, c0_arx = theta_arx.tolist()

resid_arx = y_t - X_arx @ theta_arx
sigma_arx = resid_arx.std()

print("\n=== Red Noise + BAM Forcing Model (ARX(1)) ===")
print(f"phi (AR1):        {phi_arx:.4f}")
print(f"beta (BAM coeff): {beta_arx:.4f}")
print(f"intercept c0:     {c0_arx:.4f}")
print(f"sigma (noise):    {sigma_arx:.4f}")

# simulate
y_sim_arx = np.zeros(N)
eps = rng.normal(0.0, sigma_arx, size=N)
y_sim_arx[0] = y_full[0]
for t in range(1, N):
    y_sim_arx[t] = phi_arx * y_sim_arx[t-1] + beta_arx * x_full[t] + c0_arx + eps[t]

# ======================================================
# 3) Quick comparison
# ======================================================
print("\nQuick sanity check:")
print(f"Observed ΔW mean/std:   {y_full.mean():.3f} / {y_full.std():.3f}")
print(f"RedNoise Sim mean/std:  {y_sim_rn.mean():.3f} / {y_sim_rn.std():.3f}")
print(f"ARX Sim mean/std:       {y_sim_arx.mean():.3f} / {y_sim_arx.std():.3f}")

# %%04 use the blkLWA as the threshold to define the blocking days ---------------------------

x1 = keyLWAorigin
x2 = y_sim_arx + LWA_dailyclim[:, keylon]  # add back the daily climatology
x3 = y_sim_rn + LWA_dailyclim[:, keylon]  # add back the daily climatology
a = blkLWA

# -----------------------
# Blocking event detection
# -----------------------
def run_lengths_above(series, thr):
    """Return lengths of consecutive runs where series > threshold."""
    above = series > thr
    lens, run = [], 0
    for v in above:
        if v:
            run += 1
        else:
            if run > 0:
                lens.append(run)
                run = 0
    if run > 0:
        lens.append(run)
    return np.array(lens, dtype=int)

lens1 = run_lengths_above(x1, a)
lens2 = run_lengths_above(x2, a)
lens3 = run_lengths_above(x3, a)

# Histogram of event durations
K = max(lens1.max() if len(lens1) else 1, lens2.max() if len(lens2) else 1, lens3.max() if len(lens3) else 1)
bins = np.arange(1, K + 2)  # bins for durations 1,2,3,...
counts1, _ = np.histogram(lens1, bins=bins)
counts2, _ = np.histogram(lens2, bins=bins)
counts3, _ = np.histogram(lens3, bins=bins)
centers = np.arange(1, K + 1)

# -----------------------
# Figure: Histogram of event durations (log y-axis)
# -----------------------
plt.figure(figsize=(9, 5.5))
width = 0.25
# Bar plots for the two models
plt.bar(centers, counts1, width=width, alpha=0.7, color="grey", label="Observation")
plt.bar(centers + width, counts2, width=width, alpha=0.7, color="red", label="Model: red noise + BAM")
plt.bar(centers - width, counts3, width=width, alpha=0.7, color="blue", label="Model: red noise")
# Set log scale (base e)
plt.yscale("log", base=np.e)
plt.xlabel("Blocking-event duration (days)")
plt.ylabel("Number of events (log scale, base e)")
plt.title(f"Histogram of blocking-event durations (threshold a = {a})")
plt.legend()
plt.tight_layout()
# Save before showing
plt.savefig(f"RedNoiseBAM_duration_histogram_{thX}th_lon{targetlon}.png", dpi=300)
plt.show()
plt.close()

print("Done!")
