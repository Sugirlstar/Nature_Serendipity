## main code for LWA calculation from Z500 ##
#%%
import netCDF4 as nc
import glob
import numpy as np
import xarray as xr
import LWA_f2 # LWA Calculation function
from concurrent.futures import ProcessPoolExecutor, as_completed

#%% dataset settings -------------------------------------------------------------
datasets = ["ERA5", "MERRA2", "JRA55"]

OUT_DIR_List = {
    "ERA5": "/scratch/bell/hu1029/LGHW/interm_ERA5",
    "MERRA2": "/scratch/bell/hu1029/LGHW/interm_MERRA2",
    "JRA55": "/scratch/bell/hu1029/LGHW/interm_JRA55"
}
timerefFile = {
    "ERA5": "/scratch/bell/hu1029/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc",
    "MERRA2": "/scratch/bell/hu1029/Data/processed/MERRA2_Z500_6hr_1980_2021_1dg.nc",
    "JRA55": "/scratch/bell/hu1029/Data/processed/JRA55_Z500_6hr_1979_2021_1dg.nc"
}
varnameList = {
    "ERA5": "z",
    "MERRA2": "H",
    "JRA55": "var7"
}
yearnameList = {
    "ERA5": "1979_2021",
    "MERRA2": "1980_2021",
    "JRA55": "1979_2021"
}
lat_name, lon_name, time_name = "lat", "lon", "time"

def process_one_dataset(dtname: str) -> str:

    """Compute LWA for one dataset and save outputs. Returns a status string."""
    out_dir = OUT_DIR_List[dtname]
    yearname = yearnameList[dtname]
    filepath = timerefFile[dtname]
    var_name = varnameList[dtname]
    ds0 = xr.open_dataset(filepath)
    da = ds0[var_name].squeeze() # may contian a singleton level dimension, so squeeze it out
    # ERA5: geopotential -> geopotential height (m)
    if dtname == "ERA5":
        da = da / 9.80665
    # LWA
    LWA_td, LWA_td_A, LWA_td_C, lat, lon = LWA_f2.Cal(da, lat_name, lon_name, time_name)

    # convert to lat increasing order (-90 ~ 90)
    LWA_td   = LWA_td[:, ::-1, :]
    LWA_td_A = LWA_td_A[:, ::-1, :]
    LWA_td_C = LWA_td_C[:, ::-1, :]
    lat = lat[::-1]

    # save
    np.save(f"{out_dir}/{dtname}_LWA_td_{yearname}_6hr.npy", LWA_td)
    np.save(f"{out_dir}/{dtname}_LWA_td_A_{yearname}_6hr.npy", LWA_td_A)
    np.save(f"{out_dir}/{dtname}_LWA_td_C_{yearname}_6hr.npy", LWA_td_C)
    np.save(f"{out_dir}/{dtname}_LWA_lat_{yearname}_6hr.npy", lat)
    np.save(f"{out_dir}/{dtname}_LWA_lon_{yearname}_6hr.npy", lon)

    # quick plot (save only)
    import matplotlib.pyplot as plt
    Plot = np.nanmean(LWA_td, axis=0)
    fig = plt.figure(figsize=(12, 7))
    cf = plt.contourf(lon, lat, Plot, 15, extend="both", cmap="Reds")
    plt.colorbar(cf)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    fig.savefig(f"{dtname}_LWAtotalMean_{yearname}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return f"[DONE] {dtname} -> saved to {out_dir}"


if __name__ == "__main__":
    # Use up to 3 workers (one per dataset)
    with ProcessPoolExecutor(max_workers=min(3, len(datasets))) as ex:
        futures = {ex.submit(process_one_dataset, dt): dt for dt in datasets}
        for fut in as_completed(futures):
            dt = futures[fut]
            try:
                msg = fut.result()
                print(msg, flush=True)
            except Exception as e:
                print(f"[FAIL] {dt}: {e}", flush=True)
                raise


# Serial edition:

# #%% calculate LWA and save as numpy arrays and xarray datasets ------------------------------------------------------------------------------
# for dtname in datasets:

#     OUT_DIR = OUT_DIR_List[dtname]
#     yearname = yearnameList[dtname]

#     filepath = timerefFile[dtname]
#     ds = xr.open_dataset(filepath)

#     lat_name, lon_name, time_name, var_name = 'lat', 'lon', 'time', varnameList[dtname]
#     if dtname == "ERA5":
#         ds = ds[var_name]/9.80665  # Convert to geopotential height in meters for ERA5
#     else:
#         ds = ds[var_name]  # MERRA2 and JRA55 are already in geopotential height (meters)

#     LWA_td,LWA_td_A, LWA_td_C, lat, lon = LWA_f2.Cal(ds, lat_name, lon_name, time_name)

#     # convert to lat increasing order
#     LWA_td   = LWA_td[:, ::-1, :]
#     LWA_td_A = LWA_td_A[:, ::-1, :]
#     LWA_td_C = LWA_td_C[:, ::-1, :]
#     lat = lat[::-1]

#     np.save(f"{OUT_DIR}/{dtname}_LWA_td_{yearname}_6hr.npy", LWA_td)
#     np.save(f"{OUT_DIR}/{dtname}_LWA_td_A_{yearname}_6hr.npy", LWA_td_A)
#     np.save(f"{OUT_DIR}/{dtname}_LWA_td_C_{yearname}_6hr.npy", LWA_td_C)
#     np.save(f"{OUT_DIR}/{dtname}_LWA_lat_{yearname}_6hr.npy", lat)
#     np.save(f"{OUT_DIR}/{dtname}_LWA_lon_{yearname}_6hr.npy", lon)

#     #%% Plot
#     import matplotlib.pyplot as plt
#     Plot = np.nanmean(LWA_td, axis=0)  # Sum over time to get total LWA
#     fig = plt.figure(figsize=[12,7])
#     plt.contourf(lon,lat,Plot, 15, extend="both", cmap='Reds') 
#     cb=plt.colorbar()

#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.show()
#     plt.savefig(f"{dtname}_LWAtotalMean_{yearname}.png")
#     plt.close()

#     # delete the arrays to save memory
#     del LWA_td, LWA_td_A, LWA_td_C, lat, lon

