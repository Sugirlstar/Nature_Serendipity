import numpy as np
from HYJfunction import *
import os
import gzip
import shutil
import pandas as pd
import xarray as xr
import pickle

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

#%%  extracting tracks ------------------------------------------------------------------------------

for i in range(len(datasets)):

    dtname = datasets[i]
    timerefF = timerefFile[dtname]
    print(f" -------- Processing dataset: {dtname} --------", flush=True)
    OUT_DIR = OUT_DIR_List[dtname]

    # get the time
    ds = xr.open_dataset(timerefF)
    base_dir = f'/scratch/bell/hu1029/LGHW/TRACK/{dtname}_TRACK_inputdata_geopotentialAnomaly_yearly/TRACKS/'
    folders = sorted(os.listdir(base_dir))

    # -------------- CC ----------------

    allyear_Tracks = []
    for folder in folders:

        # get the year
        yyyy = folder.split('_')[3] 
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
                print(f"Processed data for year {yyyy} -----------------", flush=True)

        allyear_Tracks.extend(updated_data)

    with open(f'{OUT_DIR}/{dtname}_CCZanom_allyearTracks_NH.pkl', 'wb') as file:
        pickle.dump(allyear_Tracks, file)  

    with open(f'{OUT_DIR}/{dtname}_CCZanom_allyearTracks_NH.pkl', 'rb') as file:
        allyearTracks = pickle.load(file)

    print(allyearTracks[0])
    print(allyearTracks[-1])

    print('done', flush=True)


    # -------------- AC ----------------    

    allyear_Tracks = []
    for folder in folders:

        # get the year
        yyyy = folder.split('_')[3] 
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

    with open(f'{OUT_DIR}/{dtname}_ACZanom_allyearTracks_NH.pkl', 'wb') as file:
        pickle.dump(allyear_Tracks, file)  

    with open(f'{OUT_DIR}/{dtname}_ACZanom_allyearTracks_NH.pkl', 'rb') as file:
        allyearTracks = pickle.load(file)

    print(allyearTracks[0])
    print(allyearTracks[-1])

    print('done', flush=True)

