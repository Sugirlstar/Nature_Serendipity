# Workflow

> This is the technical document explaining the workflow for the eddy-blocking interaction project. Apologize for the messy code and data coordinates. I will make it better!!!

## Data Coordinates
There are various datasets with different coordinates and resolutions used, espcially different order of latitude values:
1. **/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021.nc**: 6-hourly, F128, latitude decreasing (90~-90), 
2. **/Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc**: 6-hourly, 1degree, latitude **increasing (-90~90)**
3. **LWA_td_1979_2021_ERA5_6hr.npy** and **LWA_lat_1979_2021_ERA5_6hr.npy**: 6-hourly, 1degree, latitude decreasing (90~-90). 
   > the LWA is calculated based on the previous data, but its latitude has been flipped through `LWA_f2.py`, so it's decreasing.

---

## Public functions and scripts
> the slurm `JupyterDebug.slurm` is for applying the cores running jupyternotebook. Run the slurm first, then ssh to the target server, then select the interpreter in notebook in VScode, as indicated by the slurm output file.

```python
def findClosest(lati, latids):
    if isinstance(lati, np.ndarray):  # if lat is an array
        closest_indices = []
        for l in lati:  
            diff = np.abs(l - latids)
            closest_idx = np.argmin(diff) 
            closest_indices.append(closest_idx)
        return closest_indices
    else:
        # if lat is a single value
        diff = np.abs(lati - latids)
        return np.argmin(diff) 
```
---

## LWA calculation (./LWACodes)
> [code was provided by Zhaoyu, modified based on 6-hourly data, the original code is from /depot/wanglei/data/ERA5_LWA_Z500]
1. Regrid the data to 1dg: `Z500regrid.sh`
   - Input: **./Data/processed/ERA5_Z500_6hr_1979_2021_regulargrid_Float32.nc** (1440x721, lat decreasing 90 to -90, 0.5dg, 19790101-20211231, 6-hourly, varname: z, levels:1, at 500hPa)
   - Output: **./Data/processed/ERA5_Z500_6hr_1979_2021_1dg.nc** (360x181, lat increasing -90~90, 1dg, 19790101-20211231, 6-hourly, varname: z, levels:1, at 500hPa)
2. Calculate LWA: `./LWAcodes/main2.py` and the function `LWA_f2.py`
   - Output: **./LGHW/LWA_td_1979_2021_ERA5_6hr.npy**, **LWA_td_A_1979_2021_ERA5_6hr**, **LWA_td_C_1979_2021_ERA5_6hr**, **LWA_lat_1979_2021_ERA5_6hr**, and **LWA_lon_1979_2021_ERA5_6hr**. (1dg, latitude decreasing!)
3. Draft only: Transfer the LWA_td, LWA_td_A and LWA_td_C npy files to float32 and plot the mean map: /GRL_code/LWAcodes/LWA_checkPlot.py (shape: 62824, 181, 360)
4. Addition: Calculate the Z500 anomaly 1degree: `Z500anomalyCal.py`
   - Input (/scratch/bell/hu1029/Data/processed/): **ERA5_Z500_6hr_1979_2021_1dg.nc**
   - Output (/scratch/bell/hu1029/Data/processed): **/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc** (6-hourly, 1degree, lat increasing!)

---

## TRACK (./TrackCodes)
> [input data: Z500 anomaly (it's geopotential!, not geopotential height), F128 (256latx512lon), 1979-2021]
1. Prepare the TRACK package: track-TRACK-1.5.4 (put it under ./LGHW/TRACK). See environment configurations also in the doc 'Instructions on running the TRACK program_Yanjun.docx'
2. Download the input data and preprocess: 
   - `S0_ downloadERA5Z500.py`: download source data from the website 
     - outputs: **./Data/raw/ERA5_Z500_F128/ERA5_Z500_6hr_{yr}.grb**
   - `S0_combineNCbyTime_CDO.sh`: combine into one file 
     - outputs: **./Data/processed/ERA5_Z500_6hr_1979_2021_F128.nc**
   - `S1_CCZanom_getZ500anomaly.py`: calculate the Z500anomly 
     - outputs: **./Data/processed/ERA5_geopotential500_subtractseasonal_6hr_1979_2021.nc**
   - `S1_CCZanom_divideNCfiles.py`: divide into seprate years 
     - outputs: **./LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/ERA5_geopotentialAnomaly_6hr_F128_{year}.nc**
3. Run the TRACK, based on the multiprocess scripts: /`CycloneTrack_geopotentialAnom_MultiTracks.sh` and /`CycloneTrack_geopotentialAnom_MultiTracks_SH.sh`
   (see the configuration document 'Instructions on running the TRACK program_Yanjun.docx').
   - outputs: tracks for each year, stored in .gz files (e.g., **./LGHW/TRACK/TRACK_inputdata_geopotentialAnomaly/TRACKS_SH/ERA5_geopotentialAnomaly_6hr_F128_1979_zonefilt_T42_SH/ff_trs_neg.gz**)
4. TRACK results reading and orgnizing: `S2_readMultifilesTrack_NH.py` and `S2_readMultifilesTrack_SH.py`
   - outputs: **./LGHW/{trackType}Zanom_allyearTracks_{k}.pkl** ({trackType}=AC/CC, {k}=NH/SH)
5. Track trajectory point density: `S3_ERA5dipole_TrajectoryDensity_2Arrs.py`
   - outputs: **./LGHW/{trackType}trackPoints_arrayF128_{k}.npy** and **./LGHW/{trackType}trackPoints_array1dg_{k}.npy** (Bool array, mask of track points). **./LGHW/{trackType}trackPoints_TrackIDarray_{k}_F128.npy** and **./LGHW/{trackType}trackPoints_TrackIDarray_{k}_1dg.npy** (Float, storing the track id at each grid point, Increasing Lat). 

---

## Blocking and Seeding (./BlockingSeedFinding)
> The code for identifying blocking and seeding events, and selecting the target regions. NH and SH are separately calculated.
1. Blocking/Seeding tracking + peaking identification + diversity classifying: `S1_WatershedSeedBlocking_track_ERA5_daily_NH` and `S1_WatershedSeedBlocking_track_ERA5_daily_SH` 
   - Inputs (./LGHW/): **LWA_td_1979_2021_ERA5_6hr.npy**, **LWA_td_A_1979_2021_ERA5_6hr**, **LWA_td_C_1979_2021_ERA5_6hr**
   - Outputs (./LGHW/): 
     - **SD_{Seeding/Blocking}\_peaking_{date/lon/lat}_daily_{NH/SH}** (1d-list, each element represent the peaking date/lat/lon of each event, single value)
     - **SD_{Seeding/Blocking}Total_date_{NH/SH}** (list[list], each sublist is the dates of each seeding/blocking event)
     - **SD_{Seeding/Blocking}Total_label_{NH/SH}** (list[list of 2d-array], each sublist is the 2d bool masks of seeding/blocking locations (shape: 90,360; lat decreasing!))
     - **SD_{Seeding/Blocking}TypeI_{NH/SH}** (list, 1d, represent the type of each event, 1-ridge, 2-trough, 3-dipole)
     - **SD_{Seeding/Blocking}\_diversity_date_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; the dates of each event)
     - **SD_{Seeding/Blocking}\_diversity_label_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; each sublist is the 2d bool masks of seeding/blocking locations (shape: 90,360; lat decreasing!))
     - **SD_{Seeding/Blocking}\_diversity_peaking_date_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking date of each event)
     - **SD_{Seeding/Blocking}\_diversity_peaking_lat_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking lat of each event)
     - **SD_{Seeding/Blocking}\_diversity_peaking_lon_daily_{NH/SH}** (list[list of 3 types], [Ridge,Trough,Dipole]; list of the peaking lon of each event)
   - Figure Outputs: SD_{blocking/seeding}Freq_daily_1979_2021_watershed_{NH/SH}.png
2. Blocking data organization, put into the 3D-array and plot: `S2_Blocking_transfer2array.py`
   - Outputs (/scratch/bell/hu1029/LGHW/): 
     - **SD_{eve}FlagmaskClusters_Type{type_idx+1}_{rgname}_{ss}.npy** (3d array, [time, lat, lon], bool, mask of block or not; if not: 0/False)
     - **SD_{eve}FlagmaskClustersEventList_Type{type_idx+1}_{rgname}_{ss}** (1d list, saving the target region's blocking/seeding event global id)
     - **SD_{eve}ClustersEventID_Type{type_idx+1}_{rgname}_{ss}.npy** (3d array, [time, lat, lon], int, saving the blocking/seeding event's global id in the target positions; position with no event: -1)

   ### Blocking Identification Method Description
   0. Parameters for blocking and seeding:
   > dlat = dlon = 1
   BlockingDuration = 5
   SeedingDuration = 3
   BlockingLonWidth = 15
   SeedingLonWidth = 15
   valueBlockingThresh = 50  # percentile
   valueSeedingThresh = 25    # percentile
   DX_THRESH = int(18*3)
   MIN_DIST = 5
   lon_thresh = 18
   lat_thresh = 13.5

   1. get the connected area that over the threshold. Threshold: the 50th/25th of the longitude maximum 
   2. get the maximum LWA location of each individule cluster (each day) and it's area and width
      - Watershed algorithm was applied to divide big regions with multiple peak values with a longitude distance > 18*3
   3. filter the individual clusters: a. the lon width should be larger than 15 but smaller than 120; b. should not be tropical (lat>30)
   4. during the consecutive two days, find the pair events (with the shortest dististance,  and limited to 18 degree longitude and 13.5 degree latitude)
   5. start tracking the paired cluster day by day:
      5.1 find the pair event at next day
      5.2 if this event does have a pair next day, and the displacement is within 1.5*18 lons and 1.5*13.5 lats, then keep tracking
      5.3 when there are no pair events in the next day, the track is end
      5.4 a filter was applied on blocking events only (parameter in BKSDIdentifyFun: stationary=True): if the cluster center travels from the initial point more than lon_thresh*1.5 or lat_thresh*1.5, then interupt.

## BAM (./BAM)
1. Calculate the EKE: `S1_EKE_total.py` 
   - Input(/depot/wanglei/data/ERA5_uvT/): **u_component_of_wind_*.nc and v_component_of_wind_*.nc**
   - Output(/Data/processed/ERA5_EKE_total_1979_2021): **{SH/NH}_TROP_{year}.nc**
2. Get the BAM index: `S2_SBAM_index_EKE.py`
   - Output (/scratch/bell/hu1029/LGHW/): **{k}_BAM_index_total_no_leap.npy**
3. Get the high and low BAM phase: `S3_getBAMphase.py` (the BAM index has been interpolated to with 0229 since here)
   - Output: `{k}_BAM_event_peak_list.pkl` and `{k}_BAM_event_low_list.pkl` the date index of peak/low BAM state (only one day for each event)

## Eddy-Blocking Interactions (./EddyBlockingCodes)
1. Find the blocking-eddy interaction cases: `S1_blockingTrackInteraction.py` (blocking daily data are transfered to 6-hourly by repeat 4 times per day; track points are transfer to 1dg resolution by finding the nearest grid point (findClosest); Only 'through' and 'absorbed' eddies are considered as interacting eddies.)
   - Outputs (/scratch/bell/hu1029/LGHW/): 
     - **BlockingEventPersistence_Type{typeid}_{rgname}_{ss}.npy**, blocking persistence for each event, 1-d array.
     - **BlockingType{typeid}_EventEddyNumber_1979_2021_{rgname}_{ss}_{cyc}.npy**, 1-d array, the number of interacting number for each blocking event. length = len of regional blocking event. 0 for no interaction
     - **TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy**, 1-d array of the blockingid that each track is related to; length = track number. -1 for no interaction
     - **BlockingType{typeid}_ThroughTrack_1979_2021_{rgname}_{ss}_{cyc}.npy**, 1-d array, storing the 'through' track index (the global position)
     - **BlockingType{typeid}_AbsorbedTrack_1979_2021_{rgname}_{ss}_{cyc}.npy**, 1-d array, storing the 'absorbed' track index
     - **BlockingType{typeid}_EdgeTrack_1979_2021_{rgname}_{ss}_{cyc}.npy**, 1-d array, storing the 'edge' track index
     - **BlockingType{typeid}_InterType_1979_2021_{rgname}_{ss}_{cyc}.npy**, 1-d array of the interaction type that each track is related to; length = track number. 'N' for no interaction, 'T' for through, 'A' for absorbed, 'E' for edge.
  
     - **Interaction_summary.txt**, summary of length of each interaction type.
     - **BlkPersis_EddyNumber_Cor.txt**, summary of correlation between blocking persistence and eddy numbers
2. Find the first day center for each blocking event: `S2_00_getBlocking1stDayLoc.py`
   - Outputs (/scratch/bell/hu1029/LGHW/): **Blocking1stday{Date/Lat/Lon}List_blkType{typeid}_{rgname}_{ss}** (the date, lat and lon values for each first day center)
3. Calculate the density of tracks that have interacted with blockings: `S2_01_InteractingTrajectoryDensity.py`
   - Inputs(/scratch/bell/hu1029/LGHW/): **TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy** and **{cyc}Zanom_allyearTracks{HMi}.pkl**
   - Outputs (/scratch/bell/hu1029/LGHW/): 
     - **{cyc}trackInteracting_array_Type{typeid}_{rgname}_{ss}.npy**: 3d-array, bool, 1 for interacted track location, 0 for no interaction. Lat Increasing!
     - **{cyc}trackInteracting_idarr_Type{typeid}_{rgname}_{ss}.npy**: 3d-array, storing the eddy's global id at each location (the id is directly extract from the .pkl). Lat Increasing!
4. Get the composite of blocking events and related eddy track points: `S2_02_CenterComps_1stdayCenter.py`
   - Inputs (/scratch/bell/hu1029/LGHW/): **{AC/CC}trackInteracting_array_Type{typeid}_{rgname}_{ss}.npy**, **Blocking1stday{Date/Lat/Lon}List_blkType{typeid}_{rgname}_{ss}** and **/Data/processed/ERA5_Z500anomaly_subtractseasonal_6hr_1979_2021_1dg.nc** (Lat increasing)
   - Outputs (/scratch/bell/hu1029/LGHW/): **CenteredZ500_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy**, **CenteredAC_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy** and **CenteredCC_timewindow41_BlkType_Type{typeid}_{rgname}_{ss}.npy** The centered slices of all blocking events and related AC/CC track points. 3d arr (event, relativeTime, relativeLat, relativeLon).
5. Get the enter time: `S3_EnterLeavingTime.py`
   - Inputs: **{AC/CC}Zanom_allyearTracks_{NH/SH}.pkl**, **TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy**, **SD_BlockingFlagmaskClustersEventList_Type{typeid}_{rgname}_{ss}**, **Blocking1stdayDateList_blkType{typeid}_{rgname}_{ss}**
   - Outputs:
     - **EnterTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy** (the eddies' entry time, relative to blocking's 1st day; length = len of interacting blk, -1 was skipped)
     - **LeaveTimePointr2Blk1stDay_type{typeid}_{cyc}_{rgname}_{ss}.npy** (the eddies' leave time, realtive to blocking's 1st day; length = len of interacting blk, -1 was skipped)
     - **intopercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy** (the entrey time relative to the total blocking duration, percentage; length = len of interacting blk, -1 was skipped)
     - **leavepercentList_type{typeid}_{cyc}_{rgname}_{ss}.npy** (the leave time relative to the total blocking duration, percentage; length = len of interacting blk, -1 was skipped)
     > InteractingBlockID = np.load(f'/scratch/bell/hu1029/LGHW/TrackBlockingType{typeid}_Index_1979_2021_{rgname}_{ss}_{cyc}.npy') 
     > for i,blockid in enumerate(InteractingBlockID): if blockid >=0: intopercentList.append(intopercent)

## BAM and blocking seeds (./BAMwithTrack)
> BAM modulates blocking seeds - to highlight what controls the variation of the seeds. 
> growing seeds vs dying seeds - to highlight that the serendipity encounter is the key. 
> High BAM produce more seeds. Seeds encontour with eddies have higher chance to develop into blockings.


1. Figure1: High BAM v.s. Low BAM situations, 1d lines (both seeds and blocking probability)
2. Figure2: Composites of Seeds + Track points development, both under high BAM state, but one develop into blockings, another didn't develop into blockings