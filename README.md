# Nature_TechDetails

# Project Document

> This is the technical document and workflow for the serendipity theory. Including: Data description; Environment setting; TRACK program instruction; Python code for: LWA, BAM, blocking, eddy, eddy-blocking interaction analysis, and figure generating for MS and SI.
> 

<aside>
📌

make sure in slurm script:
`export PYTHONPATH=/home/hu1029/Nature_Serendipity:$PYTHONPATH`
and in notebook, add:

`import sys
sys.path.insert(0, "/home/hu1029/Nature_Serendipity")`

</aside>

---

## ⭐️ Data Description

There are three datasets with different coordinates and resolutions used, espcially different order of latitude values (-90 to 90 or 90 to -90):

### Raw data

1. **ERA5 reanalysis pressure levels, 6-houly, 0.25 degree**: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview
[variable: Geopotential (z), unit: m**2 s*-*2; format: netcdf]
2. **MERRA2**: https://disc.gsfc.nasa.gov/datasets/M2I6NPANA_5.12.4/summary?keywords=MERRA-2%20inst6_3d_ana_Np
[variable: Geopotential height (H), unit: m; format: netcdf]
3. **JRA55**: https://gdex.ucar.edu/datasets/d628000/dataaccess/#
[variable: Geopotential height (isobaric analysis field), unit: gpm; format: grib1]

### Processed data

1. ERA5
    
    
    | description | file name (under **Data/processed)** | details | generated with |
    | --- | --- | --- | --- |
    | S0: download | **/Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{year}.nc
    also copy to:
    /scratch/bell/hu1029/LGHW/TRACK/ERA5_TRACK_inputdata_geopotential_yearly** | geopotential, F128, original, 6-hourly, yearly file, Latitude decreasing (90~-90) | `dataprep_ERA5/S0_downloadERA5Z500_F128` |
    | S1: combine multiple single-year files | **ERA5_Z500_6hr_1979_2021_regulargrid_Float32.nc** | geopotential, 1440x721, lat decreasing 90 to -90, 0.25dg, 19790101-20211231, 6-hourly, varname: z, levels:1, at 500hPa | `cdo mergetime` |
    | S1: 1dg, for Blocking | **ERA5_Z500_6hr_1979_2021_1dg.nc** | geopotential, 360x181, latitude increasing (-90~90), 0-359, 6-hourly | `dataprep_ERA5/S1_Z500regrid.sh` |
    | S1: F128, for TRACK | **ERA5_Z500_6hr_1979_2021_F128.nc** | geopotential, F128, 6-hourly. Latitude increasing (-90~90), 0-359 | `dataprep_ERA5/S1_combineNCbyTime_CDO_F128.sh` |
    | S2: F128 single-year files, for TRACK | **/LGHW/TRACK/ERA5_TRACK_inputdata_geopotential_yearly** | copy from **/Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{year}.nc** |  |
    | S3: Z500 anomaly and climatology | **ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc and ERA5_Z500climatology_monthly_1979_2021_1dg.nc
    
    ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_F128.nc and ERA5_Z500climatology_monthly_1979_2021_F128.nc** | geopotential height (m), 1dg and F128, lat increasing, 0-359
    anomaly and climatology | `dataprep_ERA5/S3_Z500anomalyCal.py`
    
    also make sanity-check plots:
    **ZanomClim12Months_1dg/F128.png** |
    | S4: divide into single year | **/LGHW/TRACK/ERA5_TRACK_inputdata_geopotentialAnomaly_yearly/ERA5_geopotentialAnomaly_6hr_{year}.nc** | geopotential, F128, lat decreasing | `dataprep_ERA5/S4_divideNCfiles.sh` |
    - old version
        
        
        | description | file name | details | generated with |
        | --- | --- | --- | --- |
        | S3: 1dg, for Blocking | **ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc** | 6-hourly, 1dg, latitude decreasing (90~-90) | `dataprep_ERA5/S3_Z500anomalyCal.py` |
        | S3: F128, for TRACK | **ERA5_geopotential500_subtractseasonal_6hr_1979_2021.nc** | 6-hourly, F128, latitude decreasing (90~-90) | `dataprep_ERA5/S3_F128_getZ500anomaly.py` |
        | S4: F128 anomaly single files | **/TRACK/TRACK_inputdata_geopotentialAnomaly/ERA5_geopotentialAnomaly_6hr_F128_{year}.nc** | output_dir = '…/TRACK/TRACK_inputdata_geopotentialAnomaly'  | `dataprep_ERA5/S4_TrackZanom_divideNCfiles.py` |
        | S4: F128 geopotential 5-year files | **/TRACK/ERA5_TRACK_inputdata_geopotential/ERA5_Z500_6hr_{year1}_{year2}_F128.nc** | geopotential, 6-hourly, F128, 5-year in a file, for TRACK running | `dataprep_ERA5/S4_combineNCby5years_CDO_F128` |
        1. Calculate the anomaly
            
            
            | description | file name | details | generated with |
            | --- | --- | --- | --- |
            | S3: 1dg, for Blocking | **ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc** | geopotential height, 6-hourly, 1dg, latitude decreasing (90~-90) | `dataprep_ERA5/S3_Z500anomalyCal.py` |
            | S3: F128, for TRACK | **ERA5_geopotential500_subtractseasonal_6hr_1979_2021.nc** | geopotential, 6-hourly, F128, latitude decreasing (90~-90) | `dataprep_ERA5/S3_F128_getZ500anomaly.py` |
            
2. MERRA2
    
    
    | description | file name (under **Data/processed)** | details | generated with |
    | --- | --- | --- | --- |
    | S0: download and select vars and levels | **MERRA2/Z500/MERRA2_100.inst6_3d_ana_Np.{yyyymmdd}_Z500.nc** | geopotential height (m), lon: -180 to 179.375 by 0.625, -90 to 90 by 0.5, 6-hourly | `dataprep_MERRA2/S0_downloadPick.sh` |
    | S1: 1dg and F128 multi-year file | **MERRA2_Z500_6hr_1980_2021_1dg.nc
    and
    MERRA2_Z500_6hr_1980_2021_F128.nc** | geopotential height (m), 360x181 and F128, latitude increasing (-90~90), 6-hourly | `dataprep_MERRA2/S1_mergetime_1dg_F128.sh` |
    | S2: F128 single-year files, for TRACK | **/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotential_yearly** | geopotential, F128, 6-hourly. Latitude increasing (-90~90), 0-359 | `dataprep_MERRA2/S2_makesingleyearF128File` |
    | S3: Z500 anomaly and climatology | **MERRA2_Z500anomaly_subtractseasonal_6hr_1980_2021_1dg.nc and MERRA2_Z500climatology_monthly_1980_2021_1dg.nc
    
    MERRA2_Z500anomaly_subtractseasonal_6hr_1980_2021_F128.nc and MERRA2_Z500climatology_monthly_1980_2021_F128.nc** | geopotential height (m), 1dg and F128, lat increasing, 0-359
    anomaly and climatology | `dataprep_MERRA2/S3_Z500anomalyCal.py`
    also make sanity-check plots:
    **ZanomClim12Months_1dg/F128.png** |
    | S4: divide into single year | **/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotentialAnomaly_yearly/MERRA2_geopotentialAnomaly_6hr_{year}.nc** | geopotential, F128, lat decreasing | `dataprep_MERRA2/S4_divideNCfiles.sh` |
    
    > Some files are broken while downloading, use `check_checkFile.sh` and `check_rebuildbad.sh` to check and rebuild.
    > 
    
3. JRA55
    
    
    | description | file name (under **Data/processed)** | details | generated with |
    | --- | --- | --- | --- |
    | S1: 1dg and F128 multi-year file | **JRA55_Z500_6hr_1980_2021_1dg.nc
    and
    JRA55_Z500_6hr_1980_2021_F128.nc** | geopotential height (m), 360x181 and F128, latitude increasing (-90~90), 6-hourly | `dataprep_JRA55/S1_mergetime_1dg_F128.sh` |
    | S2: F128 single-year files, for TRACK | **/scratch/bell/hu1029/LGHW/TRACK/MERRA2_TRACK_inputdata_geopotential_yearly** | geopotential height (m), F128, 6-hourly. Latitude increasing (-90~90), 0-359 | `dataprep_JRA55/S2_makesingleyearF128File` |
    | S3: Z500 anomaly and climatology | **JRA55_Z500anomaly_subtractseasonal_6hr_1980_2021_1dg.nc and JRA55_Z500climatology_monthly_1980_2021_1dg.nc
    
    JRA55_Z500anomaly_subtractseasonal_6hr_1980_2021_F128.nc and JRA55_Z500climatology_monthly_1980_2021_F128.nc** | geopotential height (m), 1dg and F128, lat increasing, 0-359
    anomaly and climatology | `dataprep_JRA55/S3_Z500anomalyCal.py`
    also make sanity-check plots:
    **ZanomClim12Months_1dg/F128.png** |
    | S4: divide into single year | **/LGHW/TRACK/JRA55_TRACK_inputdata_geopotentialAnomaly_yearly/JRA55_geopotentialAnomaly_6hr_{year}.nc** | geopotential, F128, lat decreasing | `dataprep_JRA55/S4_divideNCfiles.sh` |
    
    > Some files are broken while downloading, use `check_checkFile.sh` and `check_redownload.sh` to check and rebuild.
    > 

<aside>
💡

There are two sets of input files for running TRACK: geopotential and geopotential anomaly. 

Currently using: geopotential anomaly

</aside>

---

## ⭐️TRACK (./TrackCodes)

> Input data: geopotential (not geopotential height), F128 (256latx512lon), 1979-2021, 6houly
For the instructions for configuring and running TRACK, please refer to ***Instructions on running the TRACK program_Yanjun.docx***
> 
1. Download and set the TRACK package: track-TRACK-1.5.4 (put it under ./LGHW/TRACK)
2. Run the TRACK, based on the multiprocess scripts: `S1_CycloneTrack_geopotential_MultiTracks.sh` and /`S1_CycloneTrack_geopotential_MultiTracks_SH.sh`
(see the configuration document ‘Instructions on running the TRACK program_Yanjun.docx’ for detailed explanation of the scripts).
    - outputs: tracks for each year, stored in .gz files (e.g., **./LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/TRACKS_SH/ERA5_geopotentialAnomaly_6hr_F128_1979_zonefilt_T42_SH/ff_trs_neg.gz**)
3. TRACK results reading and orgnizing: `S2_readMultifilesTrack_NH.py` and `S2_readMultifilesTrack_SH.py`
    - outputs: **./LGHW/{trackType}Zanom_allyearTracks_{k}.pkl** ({trackType}=AC/CC, {k}=NH/SH)
4. Track trajectory point density: `S3_ERA5dipole_TrajectoryDensity_2Arrs.py`
    - outputs: **./LGHW/{trackType}trackPoints_arrayF128_{k}.npy** and **./LGHW/{trackType}trackPoints_array1dg_{k}.npy** (Bool array, mask of track points). **./LGHW/{trackType}trackPoints_TrackIDarray_{k}_F128.npy** and **./LGHW/{trackType}trackPoints_TrackIDarray_{k}_1dg.npy** (Float, storing the track id at each grid point, Increasing Lat).
- Workflow

| description | Inputs | Script (under ./TrackCodes/) | Outputs |
| --- | --- | --- | --- |
| S1: run the TRACK program | **/scratch/bell/hu1029/LGHW/TRACK/{dataset}_TRACK_inputdata_geopotential_yearly/{dataset}_Z500_6hr_{year}.nc** | `S1_CycloneTrack_geopotential_MultiTracks.sh` and `S1_CycloneTrack_geopotential_MultiTracks_SH.sh`
 | **TRACK/{dataset}_TRACK_inputdata_geopotential_yearly/TRACKS/{dataset}_Z500_6hr_zonefilt_T42/ff_trs_neg.gz or ff_trs_pos.gz
and
TRACK/{dataset}_TRACK_inputdata_geopotential_yearly/TRACKS_SH/{dataset}_Z500_6hr_zonefilt_T42/ff_trs_neg.gz or ff_trs_pos.gz** |
| S2: read from TRACK results, turn into .pkl | last outputs | `S2_readMultifilesTrack_NH.py` and `S2_readMultifilesTrack_SH.py` | **{OUT_DIR}/{dtname}_CCZanom_allyearTracks_NH.pkl
and
{OUT_DIR}/{dtname}_CCZanom_allyearTracks_SH.pkl** |
| S3: transfer the Track to 3d-arrays | last outputs
Time reference file: /processed/{dtname}_Z500_6hr_1979_2021_1dg.nc
Coordinates reference file:
/processed/{dtname}_Z500climatology_monthly_1979_2021_F128.nc (lat increasing) | `S3_ERA5dipole_TrajectoryDensity_2Arrs` | **{OUT_DIR}/{dtname}*{trackType}trackPoints_arrayF128/1dg_*{k}.npy** : bool 3d array, 1/0 represents there are trackpoints or not
**{OUT_DIR}/{dtname}*{trackType}trackPoints_TrackIDarray_*{k}_F128/1dg.npy** : 3d array, each location store the track’s ID (0 for none)
and sanity-check plots:
**./TrackCodes/{dtname}_{trackType}*pointFrequency*{k}_F128.png**
 |

---

## ⭐️ LWA calculation (./LWACodes)

> Code was provided by Zhaoyu, modified based on 6-hourly data, the original code is from /depot/wanglei/data/ERA5_LWA_Z500
> 
- Inputs: **{dtname}_Z500_6hr_1980_2021_1dg.nc** (1dg, lat increasing order)
- Calculate LWA: `./LWAcodes/main2.py` base on the function `LWA_f2.py`
    
    ( Note: in the LWA_f2.py, the latitude of the input .nc file was forced to be decreasing (90~-90). In the `main2.py` , the results are converted to lat-increasing order.)
    
- Outputs:
    
    ```python
        np.save(f"{OUT_DIR}/{dtname}_LWA_td_{yearname}_6hr.npy", LWA_td)
        np.save(f"{OUT_DIR}/{dtname}_LWA_td_A_{yearname}_6hr.npy", LWA_td_A)
        np.save(f"{OUT_DIR}/{dtname}_LWA_td_C_{yearname}_6hr.npy", LWA_td_C)
        np.save(f"{OUT_DIR}/{dtname}_LWA_lat_{yearname}_6hr.npy", lat)
        np.save(f"{OUT_DIR}/{dtname}_LWA_lon_{yearname}_6hr.npy", lon)
    ```
    

---

## ⭐️ Public functions and scripts

> the  `JupyterDebug.slurm` is for applying the cores running jupyternotebook. Run the slurm first, then ssh to the target server, then select the interpreter in notebook in VScode, as indicated by the slurm output file.
> 

---

---

---

## Blocking and Seeding (./BlockingSeedFinding)

> The code for identifying blocking and seeding events, and selecting the target regions. NH and SH are separately calculated.
> 
1. Blocking/Seeding tracking + peaking identification + diversity classifying: `S1_WatershedSeedBlocking_track_ERA5_daily_NH` and `S1_WatershedSeedBlocking_track_ERA5_daily_SH`
    - Inputs (./LGHW/): **LWA_td_1979_2021_ERA5_6hr.npy**, **LWA_td_A_1979_2021_ERA5_6hr**, **LWA_td_C_1979_2021_ERA5_6hr**
    - Outputs (./LGHW/):
        - **SD_{Seeding/Blocking}_peaking_{date/lon/lat}*daily*{NH/SH}** (1d-list, each element represent the peaking date/lat/lon of each event, single value)
        - **SD_{Seeding/Blocking}Total_date_{NH/SH}** (list[list], each sublist is the dates of each seeding/blocking event)
        - **SD_{Seeding/Blocking}Total_label_{NH/SH}** (list[list of 2d-array], each sublist is the 2d bool masks of seeding/blocking locations (shape: 90,360; lat decreasing!))
        - **SD_{Seeding/Blocking}TypeI_{NH/SH}** (list, 1d, represent the type of each event, 1-ridge, 2-trough, 3-dipole)
        - **SD_{Seeding/Blocking}_diversity_date_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; the dates of each event)
        - **SD_{Seeding/Blocking}_diversity_label_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; each sublist is the 2d bool masks of seeding/blocking locations (shape: 90,360; lat decreasing!))
        - **SD_{Seeding/Blocking}_diversity_peaking_date_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking date of each event)
        - **SD_{Seeding/Blocking}_diversity_peaking_lat_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking lat of each event)
        - **SD_{Seeding/Blocking}_diversity_peaking_lon_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking lon of each event)
    - Figure Outputs: SD_{blocking/seeding}Freq_daily_1979_2021_watershed_{NH/SH}.png
2. Blocking data organization, put into the 3D-array and plot: `S2_Blocking_transfer2array.py`
    - Outputs (/scratch/bell/hu1029/LGHW/):
        - **SD_{eve}FlagmaskClusters_Type{type_idx+1}*{rgname}*{ss}.npy** (3d array, [time, lat, lon], bool, mask of block or not; if not: 0/False)
        - **SD_{eve}FlagmaskClustersEventList_Type{type_idx+1}*{rgname}*{ss}** (1d list, saving the target region’s blocking/seeding event global id)
        - **SD_{eve}ClustersEventID_Type{type_idx+1}*{rgname}*{ss}.npy** (3d array, [time, lat, lon], int, saving the blocking/seeding event’s global id in the target positions; position with no event: -1)
    
    ### Blocking Identification Method Description
    
    1. Parameters for blocking and seeding:
    
    > dlat = dlon = 1
    BlockingDuration = 5
    SeedingDuration = 3
    BlockingLonWidth = 15
    SeedingLonWidth = 15
    valueBlockingThresh = 50 # percentile
    valueSeedingThresh = 25 # percentile
    DX_THRESH = int(18*3)
    MIN_DIST = 5
    lon_thresh = 18
    lat_thresh = 13.5
    > 
    1. get the connected area that over the threshold. Threshold: the 50th/25th of the longitude maximum
    2. get the maximum LWA location of each individule cluster (each day) and it’s area and width
        - Watershed algorithm was applied to divide big regions with multiple peak values with a longitude distance > 18*3
    3. filter the individual clusters: a. the lon width should be larger than 15 but smaller than 120; b. should not be tropical (lat>30)
    4. during the consecutive two days, find the pair events (with the shortest dististance, and limited to 18 degree longitude and 13.5 degree latitude)
    5. start tracking the paired cluster day by day:
    5.1 find the pair event at next day
    5.2 if this event does have a pair next day, and the displacement is within 1.5*18 lons and 1.5*13.5 lats, then keep tracking
    5.3 when there are no pair events in the next day, the track is end
    5.4 a filter was applied on blocking events only (parameter in BKSDIdentifyFun: stationary=True): if the cluster center travels from the initial point more than lon_thresh*1.5 or lat_thresh*1.5, then interupt.

## BAM (./BAM)

1. Calculate the EKE: `S1_EKE_total.py`
    - Input(/depot/wanglei/data/ERA5_uvT/): **u_component_of_wind_*.nc and v_component_of_wind_*.nc**
    - Output(/Data/processed/ERA5_EKE_total_1979_2021): **{SH/NH}*TROP*{year}.nc**
2. Get the BAM index: `S2_SBAM_index_EKE.py`
    - Output (/scratch/bell/hu1029/LGHW/): **{k}_BAM_index_total_no_leap.npy**
3. Get the high and low BAM phase: `S3_getBAMphase.py` (the BAM index has been interpolated to with 0229 since here)
    - Output: `{k}_BAM_event_peak_list.pkl` and `{k}_BAM_event_low_list.pkl` the date index of peak/low BAM state (only one day for each event)

## Eddy-Blocking Interactions (./EddyBlockingCodes)

1. Find the blocking-eddy interaction cases: `S1_blockingTrackInteraction.py` (blocking daily data are transfered to 6-hourly by repeat 4 times per day; track points are transfer to 1dg resolution by finding the nearest grid point (findClosest); Only ‘through’ and ‘absorbed’ eddies are considered as interacting eddies.)
    - Outputs (/scratch/bell/hu1029/LGHW/):
        - **BlockingEventPersistence_Type{typeid}*{rgname}*{ss}.npy**, blocking persistence for each event, 1-d array.
        - **BlockingType{typeid}*EventEddyNumber_1979_2021*{rgname}*{ss}*{cyc}.npy**, 1-d array, the number of interacting number for each blocking event. length = len of regional blocking event. 0 for no interaction
        - **TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy**, 1-d array of the blockingid that each track is related to; length = track number. -1 for no interaction
        - **BlockingType{typeid}*ThroughTrack_1979_2021*{rgname}*{ss}*{cyc}.npy**, 1-d array, storing the ‘through’ track index (the global position)
        - **BlockingType{typeid}*AbsorbedTrack_1979_2021*{rgname}*{ss}*{cyc}.npy**, 1-d array, storing the ‘absorbed’ track index
        - **BlockingType{typeid}*EdgeTrack_1979_2021*{rgname}*{ss}*{cyc}.npy**, 1-d array, storing the ‘edge’ track index
        - **BlockingType{typeid}*InterType_1979_2021*{rgname}*{ss}*{cyc}.npy**, 1-d array of the interaction type that each track is related to; length = track number. ‘N’ for no interaction, ‘T’ for through, ‘A’ for absorbed, ‘E’ for edge.
        - **Interaction_summary.txt**, summary of length of each interaction type.
        - **BlkPersis_EddyNumber_Cor.txt**, summary of correlation between blocking persistence and eddy numbers
2. Find the first day center for each blocking event: `S2_00_getBlocking1stDayLoc.py`
    - Outputs (/scratch/bell/hu1029/LGHW/): **Blocking1stday{Date/Lat/Lon}List_blkType{typeid}*{rgname}*{ss}** (the date, lat and lon values for each first day center)
3. Calculate the density of tracks that have interacted with blockings: `S2_01_InteractingTrajectoryDensity.py`
    - Inputs(/scratch/bell/hu1029/LGHW/): **TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy** and **{cyc}Zanom_allyearTracks{HMi}.pkl**
    - Outputs (/scratch/bell/hu1029/LGHW/):
        - **{cyc}trackInteracting_array_Type{typeid}*{rgname}*{ss}.npy**: 3d-array, bool, 1 for interacted track location, 0 for no interaction. Lat Increasing!
        - **{cyc}trackInteracting_idarr_Type{typeid}*{rgname}*{ss}.npy**: 3d-array, storing the eddy’s global id at each location (the id is directly extract from the .pkl). Lat Increasing!
4. Get the composite of blocking events and related eddy track points: `S2_02_CenterComps_1stdayCenter.py`
    - Inputs (/scratch/bell/hu1029/LGHW/): **{AC/CC}trackInteracting_array_Type{typeid}*{rgname}*{ss}.npy**, **Blocking1stday{Date/Lat/Lon}List_blkType{typeid}*{rgname}*{ss}** and **/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc** (Lat increasing)
    - Outputs (/scratch/bell/hu1029/LGHW/): **CenteredZ500_timewindow41_BlkType_Type{typeid}*{rgname}*{ss}.npy**, **CenteredAC_timewindow41_BlkType_Type{typeid}*{rgname}*{ss}.npy** and **CenteredCC_timewindow41_BlkType_Type{typeid}*{rgname}*{ss}.npy** The centered slices of all blocking events and related AC/CC track points. 3d arr (event, relativeTime, relativeLat, relativeLon).
5. Get the enter time: `S3_EnterLeavingTime.py`
    - Inputs: **{AC/CC}Zanom_allyearTracks_{NH/SH}.pkl**, **TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy**, **SD_BlockingFlagmaskClustersEventList_Type{typeid}*{rgname}*{ss}**, **Blocking1stdayDateList_blkType{typeid}*{rgname}*{ss}**
    - Outputs:
        - *EnterTimePointr2Blk1stDay_type{typeid}*{cyc}*{rgname}_{ss}.npy** (the eddies’ entry time, relative to blocking’s 1st day; length = len of interacting blk, -1 was skipped)
        - *LeaveTimePointr2Blk1stDay_type{typeid}*{cyc}*{rgname}_{ss}.npy** (the eddies’ leave time, realtive to blocking’s 1st day; length = len of interacting blk, -1 was skipped)
        - *intopercentList_type{typeid}*{cyc}*{rgname}_{ss}.npy** (the entrey time relative to the total blocking duration, percentage; length = len of interacting blk, -1 was skipped)
        - *leavepercentList_type{typeid}*{cyc}*{rgname}_{ss}.npy** (the leave time relative to the total blocking duration, percentage; length = len of interacting blk, -1 was skipped)
        
        > InteractingBlockID = np.load(f’/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}*Index_1979_2021*{rgname}*{ss}*{cyc}.npy’)
        for i,blockid in enumerate(InteractingBlockID): if blockid >=0: intopercentList.append(intopercent)
        > 

## BAM and blocking seeds (./BAMwithTrack)

> BAM modulates blocking seeds - to highlight what controls the variation of the seeds. 
growing seeds vs dying seeds - to highlight that the serendipity encounter is the key. 
High BAM produce more seeds. Seeds encontour with eddies have higher chance to develop into blockings.
> 
1. Figure1: High BAM v.s. Low BAM situations, 1d lines (both seeds and blocking probability)
2. Figure2: Composites of Seeds + Track points development, both under high BAM state, but one develop into blockings, another didn’t develop into blockings