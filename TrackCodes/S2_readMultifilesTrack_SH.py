import numpy as np
from HYJfunction import *
import os
import gzip
import shutil
import pandas as pd
import xarray as xr
import pickle

# get the time
ds = xr.open_dataset('/scratch/bell/hu1029/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc')
base_dir = '/scratch/bell/hu1029/LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/TRACKS_SH/'
folders = sorted(os.listdir(base_dir))

# -------------- CC ----------------    

allyear_Tracks = []
for folder in folders:

    # get the year
    yyyy = folder.split('_')[4] 
    # get all the timestamps
    yearstamp = ds['time'].sel(time=yyyy, drop=False)
    yearstamp = pd.to_datetime(yearstamp.values)

    folder_path = os.path.join(base_dir, folder)
    
    if os.path.isdir(folder_path):
        gz_file_path = os.path.join(folder_path, 'ff_trs_neg.gz')
        if os.path.exists(gz_file_path):
            # unzip
            output_file_path = os.path.join(folder_path, 'ff_trs_neg')
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            data = readTRACKoutput_version152(output_file_path)

            updated_data = []
            for i in range(len(data)):
                num_index, time_lon_lat_list = data[i]
                updated_num_index = num_index + (float(yyyy) * 0.0001)
                updated_time_lon_lat_list = []
                for time, lon, lat in time_lon_lat_list:
                    updated_time = yearstamp[time-1]
                    updated_time_lon_lat_list.append((updated_time, lon, lat))
                updated_data.append((updated_num_index, updated_time_lon_lat_list))
            print(f"Processed data for year {yyyy} -----------------")

    allyear_Tracks.extend(updated_data)

with open('/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks_SH.pkl', 'wb') as file:
    pickle.dump(allyear_Tracks, file)  

with open('/scratch/bell/hu1029/LGHW/CCZanom_allyearTracks_SH.pkl', 'rb') as file:
    allyearTracks = pickle.load(file)

print(allyearTracks[0])
print(allyearTracks[-1])

print('done')


# -------------- AC ----------------    

allyear_Tracks = []
for folder in folders:

    # get the year
    yyyy = folder.split('_')[4] 
    # get all the timestamps
    yearstamp = ds['time'].sel(time=yyyy, drop=False)
    yearstamp = pd.to_datetime(yearstamp.values)

    folder_path = os.path.join(base_dir, folder)
    
    if os.path.isdir(folder_path):
        gz_file_path = os.path.join(folder_path, 'ff_trs_pos.gz')
        if os.path.exists(gz_file_path):
            # unzip
            output_file_path = os.path.join(folder_path, 'ff_trs_pos')
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            data = readTRACKoutput_version152(output_file_path)

            updated_data = []
            for i in range(len(data)):
                num_index, time_lon_lat_list = data[i]
                updated_num_index = num_index + (float(yyyy) * 0.0001)
                updated_time_lon_lat_list = []
                for time, lon, lat in time_lon_lat_list:
                    updated_time = yearstamp[time-1]
                    updated_time_lon_lat_list.append((updated_time, lon, lat))
                updated_data.append((updated_num_index, updated_time_lon_lat_list))
            print(f"Processed data for year {yyyy} -----------------")

    allyear_Tracks.extend(updated_data)

with open('/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks_SH.pkl', 'wb') as file:
    pickle.dump(allyear_Tracks, file)  

with open('/scratch/bell/hu1029/LGHW/ACZanom_allyearTracks_SH.pkl', 'rb') as file:
    allyearTracks = pickle.load(file)

print(allyearTracks[0])
print(allyearTracks[-1])

print('done')

