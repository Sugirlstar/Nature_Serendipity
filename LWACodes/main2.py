## main code for LWA calculation from Z500 ##
#%%
import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import LWA_f2 # LWA Calculation function

filepath = "/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc"
ds = xr.open_dataset(filepath)

lat_name, lon_name, time_name, var_name = 'lat', 'lon', 'time', 'z'
ds = ds[var_name]/9.80665  # Convert to geopotential height in meters

#%%
LWA_td,LWA_td_A, LWA_td_C, lat, lon = LWA_f2.Cal(ds, lat_name, lon_name, time_name)

np.save("/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr.npy", LWA_td)
np.save("/scratch/bell/hu1029/LGHW/LWA_td_A_1979_2021_ERA5_6hr.npy", LWA_td_A)
np.save("/scratch/bell/hu1029/LGHW/LWA_td_C_1979_2021_ERA5_6hr.npy", LWA_td_C)
np.save("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy", lat)
np.save("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy", lon)

ds_LWA = xr.Dataset(
    {
        "LWA_td": (["time", "lat", "lon"], LWA_td),  
        "LWA_td_A": (["time", "lat", "lon"], LWA_td_A),  
        "LWA_td_C": (["time", "lat", "lon"], LWA_td_C)  
    },
    coords={
        "time": ds[time_name],  
        "lat": lat,             
        "lon": lon              
    },
    attrs={
        "description": "LWA calculated from Z500",  
    }
)

#%% Plot
import matplotlib.pyplot as plt
Plot = np.nanmean(LWA_td, axis=0)  # Sum over time to get total LWA
fig = plt.figure(figsize=[12,7])
plt.contourf(lon,lat,Plot, 50, extend="both", cmap='Reds') 
cb=plt.colorbar()

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
plt.savefig('LWAtotalMean_test.png')
plt.close()

