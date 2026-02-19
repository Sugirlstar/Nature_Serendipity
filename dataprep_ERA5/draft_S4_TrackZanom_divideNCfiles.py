import xarray as xr
import os

input_file = '/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_F128.nc'
output_dir = '/scratch/bell/hu1029/LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly' 

ds = xr.open_dataset(input_file)
os.makedirs(output_dir, exist_ok=True)
time = ds['time'].to_index()

for year in range(time.year.min(), time.year.max() + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        ds_new = ds.sel(time=slice(start_date, end_date))

        if len(ds_new['time']) > 0:
            output_file = os.path.join(output_dir, f"ERA5_geopotentialAnomaly_6hr_F128_{year}.nc")
            ds_new.to_netcdf(output_file)
            print(f"Exported: {output_file}")

print("All files have been exported successfully.")


