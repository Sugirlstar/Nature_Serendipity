#%%
###### This code is to reproduce Thompson's BAM result, originally from Zhaoyu ######
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
import os

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
end_date = datetime(2022, 1, 1)  
print(start_date, flush=True)

nday = (end_date - start_date).days
dates = [start_date + timedelta(days=i*1.0) for i in np.arange(nday)]
Date = pd.DataFrame({'date': pd.to_datetime(dates)})
Month = Date['date'].dt.month 
Year = Date['date'].dt.year
Day = Date['date'].dt.day


for kk in ['SH','NH']:

    #%% Step 1: Read the pre-calculated EKE data and calculate the zonal mean
    # get the directory of each nc data ###
    files = glob.glob(rf"/scratch/bell/hu1029/Data/processed/ERA5_EKE_total_1979_2021/{kk}*.nc")
    print(files, flush=True)
    files.sort()
    N = len(files)   #The number of u nc files

    ### read any data file to read some basic variables like lon and lat ###
    fil = Dataset(files[0],'r')
    lon = fil.variables['lon'][:]
    lat = fil.variables['lat'][:]
    lev0 = fil.variables['level'][:]
    fil.close()

    if kk == 'NH':
        ilat = np.where((lat >= 20) & (lat <= 70))[0]
    else:
        ilat = np.where((lat >= -70) & (lat <= -20))[0]
    ilev = np.where((lev0 >= 200) & (lev0 <= 1000))[0]
    print(f"{kk} lats:", flush=True)
    print(ilat, flush=True)
    print(ilev, flush=True)

    lev = lev0[ilev]     ### 1000-200hPa
    lat = lat[ilat]   ### 20-70 N or 20-70 S
    print('the real lat values:', flush=True)
    print(lat, flush=True)
    nlev = len(lev)
    nlat = len(lat)
    nlon = len(lon)
    print('nlev, nlat, nlon:', flush=True)
    print(nlev, nlat, nlon, flush=True)
    #%%
    ### Calculate or read the zonal-mean EKE (20-70N, 1000hPa-200hPa) ###
    # check if the zonal mean EKE data exists
    if not os.path.exists(f'/scratch/bell/hu1029/Data/processed/ERA5_EKE_ZM_{kk}_total_1979_2021_1dg.npy'):
        print('Calculating the zonal mean EKE ------------------------------', flush=True)
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
            
        np.save(f'/scratch/bell/hu1029/Data/processed/ERA5_EKE_ZM_{kk}_total_1979_2021_1dg.npy',EKE)
    else:
        EKE = np.load(f'/scratch/bell/hu1029/Data/processed/ERA5_EKE_ZM_{kk}_total_1979_2021_1dg.npy')

    # Check if any Inf exists
    print("EKE shape:", EKE.shape, flush=True)
    has_nan = np.isnan(EKE).any()
    print("Does EKE contain Inf?", has_nan, flush=True)

    # %% Step2: Calculate the BAM index following TB14
    n_season = 43 # total years
    season_day = 90
    n_season_day = n_season * season_day
    days_per_year = 365
    n_all_day= n_season * days_per_year 

    ###### Calculate the climatology ######
    ### Daiy Climatology, it should be a 365-day array, this is to remove the seaoanal cycle ###

    # Find indices where the date is February 29
    feb_29_ind = Date[(Month == 2) & (Day == 29)].index

    print(feb_29_ind, flush=True)
    # Remove February 29 from the Date DataFrame
    print(Date.shape, flush=True)
    Date_leap = Date.drop(index=feb_29_ind)
    print(Date_leap.shape, flush=True)

    # check if the reshape is correct
    Date_reshaped = np.array(Date_leap['date']).reshape(n_season, days_per_year)
    print(Date_reshaped[:,0], flush=True)

    print(EKE.shape, flush=True)
    # Remove the corresponding slices from the 3D array
    EKE = np.delete(EKE, feb_29_ind, axis=0)
    print(EKE.shape, flush=True)

    EKE_4d = EKE.reshape(n_season, days_per_year, nlev, nlat) 

    EKE_day_clim = EKE_4d.mean(axis=0)
    print(EKE_day_clim.shape, flush=True)
    
    ## Transform to frequency domain and take the first four harmonic ##
    k = np.arange(4)
    w = np.hanning(365)    ##Hanning window is required
    EKE_day_clim_new = np.zeros((days_per_year,nlev,nlat))

    for iz in np.arange(nlev):
        for iy in np.arange(nlat):
            tmp = EKE_day_clim[:,iz,iy]
            # y_a = y0 - y0.mean()
            EKE_day_clim_new[:,iz,iy] = bandpass(tmp,k,w) 

    ## plot the new curve, it should be more smooth ##
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(np.arange(365),EKE_day_clim_new[:,10,0])  
    plt.show()
    plt.savefig(f'{kk}_EKE_day_clim_new.png')
    plt.close()

    ### Climatology

    EKE_clim = np.mean(EKE_day_clim,axis=0)
    print(EKE_clim.shape, flush=True)

    ### plot the climatology ###
    maxlevel = EKE_clim.max()
    minlevel = EKE_clim.min() 
    levs = np.linspace(minlevel, maxlevel, 11)

    print(minlevel, flush=True)
    print(maxlevel, flush=True)

    xx, yy = np.meshgrid(lat, lev)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.contour(xx,yy, EKE_clim, levs, widths=1, colors='k')  #_r is reverse
    cn = plt.contourf(xx,yy, EKE_clim, levs, cmap = 'Reds')
    cbar = plt.colorbar(cn, ax=ax, orientation="horizontal",extend='both',shrink=0.9, aspect=50*0.9,drawedges='bool',pad=0.15)
    if kk == 'NH':
        plt.xlim(20,70)
    else:
        plt.xlim(-70,-20)
    plt.ylim(1000,200)
    plt.xlabel('latitude',fontsize=12)
    plt.ylabel('pressure level',fontsize=12)
    plt.title("Zonal Mean EKE Climatology", pad=20)
    plt.show()
    plt.savefig(f'{kk}_Zonal_Mean_EKE_Climatology.png')
    plt.close()


    #%%
    ###### calculate the anomaly ######        
    ### For the full years EKE, we need to exclude 02/29, so 365*43 = 15695 days in total ###
    EKE_anom = np.zeros((n_all_day,nlev,nlat))
    for t in np.arange(n_all_day):
            d = t % 365
            EKE_anom[t,:,:] = EKE[t,:,:] - EKE_day_clim_new[d,:,:]

    ###### EOF Analysis ######
    # Create an EOF solver to do the EOF analysis. Square-root of cosine of
    # latitude and mass (level) weights are applied before the computation of EOFs.
    coslat = np.cos(np.deg2rad(lat)).clip(0., 1.) #clip limits the arary to a range
    lev_diff = np.zeros(nlev)
    for k in np.arange(nlev):
        lev_diff[k] = lev0[k] - lev0[k+1]
    #lev_diff = lev_diff
    wgts = np.sqrt(coslat)[np.newaxis,:] * lev_diff[:,np.newaxis]

    solver = Eof(EKE_anom, weights = wgts)
    # Retrieve the leading EOF, expressed as the covariance between the leading PC
    # time series and the input SLP anomalies at each grid point.
    eof = solver.eofsAsCovariance(neofs=1)
    pc1 = solver.pcs(npcs=1, pcscaling=1)
    var = solver.varianceFraction()
    np.save(f'/scratch/bell/hu1029/LGHW/{kk}_BAM_index_total_no_leap.npy',pc1)
    print('BAMindex shape:', pc1.shape, flush=True)

    ### Plot the EOF pattern ###
    maxlevel = eof[0,:,:].max()
    minlevel = eof[0,:,:].min()  
    levs = np.linspace(minlevel, maxlevel, 11)

    print(minlevel, flush=True)
    print(maxlevel, flush=True)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.contour(xx,yy, eof[0,:,:], levs, widths=1, colors='k')  #_r is reverse
    cn = plt.contourf(lat,lev, eof[0,:,:], levs, cmap = 'Reds')
    cbar = plt.colorbar(cn, ax=ax, orientation="horizontal",extend='both',shrink=0.9, aspect=50*0.9,drawedges='bool',pad=0.15)
    if kk == 'NH':
        plt.xlim(20,70)
    else:
        plt.xlim(-70,-20)
    plt.ylim(1000,200)
    plt.xlabel('latitude',fontsize=12)
    plt.ylabel('pressure level',fontsize=12)
    plt.title("Leading mode of the zonal-mean EKE", pad=10)
    plt.show()
    plt.savefig(f'{kk}_LeadingMode_zonalmeanEKE.png')
    plt.close()

    #%%
    ###### The full year power spectral with Thompson's method ######
    ### Each subset includes 500 days, and the overlap between adjacent subsets is 250 days ###
    i = 0
    n_subset = 0
    pc = []
    while i+500 < n_all_day:
        pc.append(pc1[i : i+500])
        i+=250
        n_subset+=1
    ##Initiate the power spectra##
    Fs = 1
    Ts = 1.0/Fs
    t = np.arange(0,500,Ts)

    n = 500 #length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T
    frq1 = frq[range(int(n/2))] # one side frequency range

    w = np.hanning(n)

    S = np.zeros((n_subset,int(n/2)))
    for i in np.arange(n_subset):
        y = np.squeeze(pc[i])   

        A0 = np.fft.fft(y*w)
        A = np.abs(A0)**2 / n
        A1 = A[range(int(n/2))]
        S[i,:] = A1

    S_m = S.mean(axis=0)

    for j in np.arange(int(n/2)):
        if j == 0:
            S_m[j] = 0.5*S_m[0] + 0.5*S_m[1]
        elif j == int(n/2)-1:
            S_m[j] = 0.5*S_m[j] + 0.5*S_m[j-1]
        else:
            S_m[j] = 0.25*S_m[j-1] + 0.5*S_m[j] + 0.25*S_m[j+1]
            
            
    ### plot the spectral ###
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    plt.plot(frq1, S_m, '-k', linewidth=2)
    plt.title("PC1 power spectral", pad=10)
    plt.xlabel('Frequency',fontsize=12)
    plt.ylabel('Power',fontsize=12)
    plt.xticks([0.04,0.1,0.15,0.2,0.25])
    ax.set_xlim(0.01,0.25)
    ax.set_ylim(0,3)
    plt.vlines(0.04,0,3,colors='r',linestyles='solid', linewidth=1)
    plt.show()
    plt.savefig(f'{kk}_PC1powerSpectral.png')
    plt.close()

    ### plot the PC curve itself ###
    ### DJF ###
    pc = []
    for i in np.arange(n_season-1):
        pc.append(pc1[334+i*365:424+i*365])
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.plot(np.arange(90), pc[-1][:,0], '-k', linewidth=2)
    plt.title("PC1", pad=10)
    plt.xlabel('Days',fontsize=12)
    plt.ylabel('PC1',fontsize=12)
    plt.xticks([0,29,59,89])
    ax.set_xticklabels([1,30,60,90])
    plt.show()
    plt.savefig(f'{kk}_PC1_DJF.png')
    plt.close()


