#%%
###### This code is to track all blocking events ######
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime as dt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pandas as pd
import cv2
import copy
import matplotlib.path as mpath
import pickle
import glob
from netCDF4 import Dataset
import xarray as xr

import sys
sys.stdout.reconfigure(line_buffering=True) # print at once in slurm

### A function to calculate distance between two grid points on earth ###
from math import radians, cos, sin, asin, sqrt
 
def haversine(lon1, lat1, lon2, lat2): # 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # transform decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine equation
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # earth radius
    return c * r * 1000

#%%
### Read the Z500-based LWA ###
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
LWA_td = np.load("/scratch/bell/hu1029/LGHW/LWA_td_1979_2021_ERA5_6hr_float32.npy") 
# daily averaged data:
T = LWA_td.shape[0]
T4 = T // 4
LWA_td = LWA_td[:T4*4]
LWA_td = LWA_td.reshape(T4, 4, len(lat), len(lon)).mean(axis=1)

print(LWA_td.shape)
lat_mid = int(len(lat)/2) + 1 #91
lat_NH = lat[0:lat_mid-1]
print(lat_NH)
print(lon)
nlon = len(lon)
nlat = len(lat)
nlat_NH =len(lat_NH)
LWA_Z = LWA_td[:,0:lat_mid-1,:] # NH only!

### Time Management ###
Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
Date = list(Date0['date'])
nday = len(Date)

#%%
###### Detect Blocking ######
Blocking_total_lat = []
Blocking_total_lon = []
Blocking_total_date = []
B_freq = np.zeros((nlat_NH, nlon))         ### The final blocking frequency/total number 
B_freq2 = np.zeros((nday ,nlat_NH, nlon))  ### Record whether a grid point is under a blocking for a certain day (only 1 and 0 in this 3D array)

### Threshold and Duration ###
LWA_max_lon = np.zeros((nlon*nday))
for t in np.arange(nday):    
    for lo in np.arange(nlon):
        LWA_max_lon[t*nlon+lo] = np.max(LWA_Z[t,:,lo])
Thresh = np.median(LWA_max_lon[:])
Duration = 2
print('Threshold:', Thresh)

### Wave Event ###
WE = np.zeros((nday,nlat_NH,nlon),dtype='uint8') 
WE[LWA_Z>Thresh] = 255                    # Wave event

################### connected component-labeling algorithm ################
num_labels = np.zeros(nday)
labels = np.zeros((nday,nlat_NH,nlon))
for d in np.arange(nday):
    num_labels[d], labels[d,:,:], stats, centroids  = cv2.connectedComponentsWithStats(WE[d,:,:], connectivity=4)

####### connect the label around 180, since they are labeled separately ########
####### but actually they should belong to the same label  ########
labels_new = copy.copy(labels)
for d in np.arange(nday):
    if np.any(labels_new[d,:,0]) == 0 or np.any(labels_new[d,:,-1]) == 0:   ## If there are no events at either -180 or 179.375, then we don't need to do any connection
        continue
            
    column_0 = np.zeros((nlat_NH,3))       ## We assume there are at most three wave events at column 0 (-180) (actuaaly most of the time there is just one)
    column_end = np.zeros((nlat_NH,3))
    label_0 = np.zeros(3)
    label_end = np.zeros(3)
    
    ## Get the wave event at column 0 (0) ##
    start_lat0 = 0
    for i in np.arange(3):
        for la in np.arange(start_lat0, nlat_NH):
            if labels_new[d,la,0]==0:
                continue
            if labels_new[d,la,0]!=0:
                label_0[i]=labels_new[d,la,0]
                column_0[la,i]=labels_new[d,la,0]
                if labels_new[d,la+1,0]!=0:
                    continue
                if labels_new[d,la+1,0]==0:
                    start_lat0 = la+1
                    break 

        ## Get the wave event at column -1 (359) ## 
        start_lat1 = 0
        for j in np.arange(3):
            for la in np.arange(start_lat1, nlat_NH):
                if labels_new[d,la,-1]==0:
                    continue
                if labels_new[d,la,-1]!=0:
                    label_end[j]=labels_new[d,la,-1]
                    column_end[la,j]=labels_new[d,la,-1]
                    if labels_new[d,la+1,-1]!=0:
                        continue
                    if labels_new[d,la+1,-1]==0:
                        start_lat1 = la+1
                        break                       
            ## Compare the two cloumns at 0 and 359, and connect the label if the two are indeed connected
            if (column_end[:,i]*column_0[:,j]).mean() == 0:
                continue                
            if (column_end*column_0).mean() != 0:
                num_labels[d]-=1
                if label_0[i] < label_end[j]:
                    labels_new[d][labels_new[d]==label_end[j]] = label_0[i]
                    labels_new[d][labels_new[d]>label_end[j]] = (labels_new[d]-1)[labels_new[d]>label_end[j]]            
                if label_0[i] > label_end[j]:
                    labels_new[d][labels_new[d]==label_0[i]] = label_end[j]
                    labels_new[d][labels_new[d]>label_0[i]] = (labels_new[d]-1)[labels_new[d]>label_0[i]]

print('labels_new done')

############ Now we get the maximum LWA location of each individule event ###########
############ Also we get the area or width of each individual event #########
lat_d = []; lon_d = []; max_d = []
lon_w = []; lon_e = []; area = []
lon_wide = []
for d in np.arange(nday):
    if int(num_labels[d]-1)==0:
        lat_list=np.zeros((1));lon_list=np.zeros((1))
        lat_list[0] = np.nan; lon_list[0] = np.nan
        lat_d.append(lat_list)
        lon_d.append(lon_list)
        lon_w.append(lon_list)
        lon_e.append(lon_list)
        area.append(lon_list)
        lon_wide.append(lon_list)
        continue
    
    lat_list=np.zeros(( int(num_labels[d]-1) ));    lon_list=np.zeros(( int(num_labels[d]-1) ));   max_list=np.zeros(( int(num_labels[d]-1) ))
    lon_w_list = np.zeros(( int(num_labels[d]-1) ));lon_e_list=np.zeros(( int(num_labels[d]-1) )); area_list = np.zeros((int(num_labels[d]-1))) 
    lon_wide_list = np.zeros(( int(num_labels[d]-1) ))
    for n in np.arange(0,int(num_labels[d]-1)):
        LWA_d = np.zeros((nlat_NH, nlon))
        LWA_d[labels_new[d]==n+1]=LWA_Z[d][labels_new[d]==n+1]   ### isolate that wave event ###
        ### Get the maximum location ###
        if len(np.array(np.where( LWA_d==LWA_d.max() ))[0])>1:           
            lat_list[n] = lat_NH[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0][0]]
            lon_list[n] = lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1][0]]
            max_list[n] = LWA_d.max()
        else:
            lat_list[n] = lat_NH[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[0]]
            lon_list[n] = lon[np.squeeze(np.array(np.where( LWA_d==LWA_d.max() )))[1]]
            max_list[n] = LWA_d.max()
        ### Get the west east boundary longitude and area ###
        for lo in np.arange(nlon):
            if (np.any(LWA_d[:,lo])) and (not np.any(LWA_d[:,lo-1])):
                lon_w_list[n] = lon[lo]
            if (not np.any(LWA_d[:,lo])) and (np.any(LWA_d[:,lo-1])):
                lon_e_list[n] = lon[lo-1]
        ### Get the width from west to east ###
        if lon_e_list[n]-lon_w_list[n] > 0:
            lon_wide_list[n] = lon_e_list[n]-lon_w_list[n]
        else:
            lon_wide_list[n] = 360+(lon_e_list[n]-lon_w_list[n])
        ### Get the total area ###
        area_count = np.zeros((nlat_NH, nlon))
        area_count[labels_new[d]==n+1]=1
        area_list[n] = np.sum(area_count)
                    
    lat_d.append(lat_list);   lon_d.append(lon_list);    max_d.append(max_list)
    lon_w.append(lon_w_list); lon_e.append(lon_e_list);  area.append(area_list)
    lon_wide.append(lon_wide_list)

print('area get done')

#%%            
########### during the consecutive two days, find the pair events (with the shortest dististance) #############
########### the distance is limited to 18 degree longitude and 13.5 degree latitude #########
next_index = []
lon_thresh = 18
lat_thresh = 13.5
for d in np.arange(nday-1):
    next_index_day = np.full(len(lon_d[d]), np.nan)
    ### create a matrix that contains the distance between each WE during the consective two days ###
    shift_L = np.zeros((len(lon_d[d]),len(lon_d[d+1]) ))
    for i in np.arange(len(lon_d[d])):
        for j in np.arange(len(lon_d[d+1])):
            shift_L[i,j] = haversine(lon_d[d][i], lat_d[d][i], lon_d[d+1][j], lat_d[d+1][j])    
    
    ### pair the events from the shortest distance among them ###
    for dd in np.arange( min(len(lon_d[d]), len(lon_d[d+1])) ):
        WE_i, WE_j = np.unravel_index(np.argmin(shift_L), shift_L.shape)
        
        ### note the distance between two paris is also limited a thrshold ###
        lon_shift = abs(lon_d[d+1][WE_j] - lon_d[d][WE_i])
        lat_shift = abs(lat_d[d+1][WE_j] - lat_d[d][WE_i])
        ### correct the longitude shift due the periodic boundary ###        
        if lon_shift > 180:
            lon_shift = lon_shift-360
        if lon_shift < -180:
            lon_shift = lon_shift + 360
            
        if lon_shift<lon_thresh and lat_shift<lat_thresh:
            next_index_day[WE_i] = WE_j  ### these two events can be paired! ###
            shift_L[WE_i, :] = np.inf    ### make the distance related to these two events to infinity so that we can search the next shortest distance and avoid this pair ###
            shift_L[:, WE_j] = np.inf
        else:
            shift_L[WE_i, :] = np.inf
            shift_L[:, WE_j] = np.inf
            
    next_index.append(next_index_day)

print('distance filter done -------------- ')

#%%
########### Now we begin to track wave events #########
Blocking_lat = []; Blocking_lon = []; Blocking_date = []
Blocking_lon_wide = []; Blocking_area = []; Blocking_label = []
        
for d in np.arange(nday-1):
    ### if all wave events within this day are tracked, then go to the next day ###
    if np.all(np.isnan(lon_d[d])):
        continue  
            
    for i in np.arange(len(lon_d[d])):
        ### if this wave evnet is tracked, then go to the next wave event ###
        if np.isnan(lon_d[d][i]):
            continue
        ### tracking starts ###
        day = 0
        track_lon = []; track_lat = []; track_date = []
        track_lon_index = []; track_lat_index = []
        track_lon_wide = []; track_area = []; track_lat_wide = []; track_label = []
        B_count = np.zeros((nlat_NH, nlon))
        B = np.zeros((nlat_NH, nlon))
        B_count2 = []
        
        B_count[labels_new[d+day]==i+1]+=1
        B[labels_new[d+day]==i+1]=1            
        B_count2.append(B)
        
        track_lon.append(lon_d[d+day][i]); track_lon_index.append(i)               
        track_lat.append(lat_d[d+day][i]); track_lat_index.append(i)            
        track_date.append(Date[d+day])
        track_lon_wide.append(lon_wide[d+day][i])
        track_area.append(area[d+day][i])
        track_label.append(labels_new[d+day]==i+1)
        
        next_index_pair= next_index[d+day][i] ### find the pair event at next day, it could be nan ###
        if ~np.isnan(next_index_pair):
            next_index_pair = int(next_index_pair)
            
        while ~np.isnan(next_index_pair) and abs(lon_d[d+day+1][next_index_pair]-track_lon[0]) < 1.5*18 and abs(lat_d[d+day+1][next_index_pair]-track_lat[0]) < 1.5*13.5:
            ### if this event does have a pair next day, and the displacement is within 1.5*18 lons and 1.5*13.5 lats, then keep tracking ###
            track_date.append(Date[d+day+1])
            B_count[labels_new[d+day+1]==next_index_pair+1]+=1
            B = np.zeros((nlat_NH, nlon))
            B[labels_new[d+day+1]==next_index_pair+1]=1            
            B_count2.append(B)
                
            track_lon.append(lon_d[d+day+1][next_index_pair])
            track_lon_index.append(next_index_pair)
            track_lat.append(lat_d[d+day+1][next_index_pair])
            track_lat_index.append(next_index_pair)
            track_lon_wide.append(lon_wide[d+day+1][next_index_pair])
            track_area.append(area[d+day+1][next_index_pair])
            track_label.append(labels_new[d+day+1]==next_index_pair+1)
            
            ### interate the day ###
            day+=1
            
            ### if this the last day, then jump out ###
            if d+day+1>nday-1:
                break
            
            ### if not, then find a next pair ###
            next_index_pair= next_index[d+day][next_index_pair]
            if ~np.isnan(next_index_pair):
                next_index_pair = int(next_index_pair)
        
            
        ### when there are no pair events in the next day, the track is end ###
        ### make sure the wave event is large enough ###
        n_large_wave = 0
        for j in np.arange(len(track_lon_wide)):
            if track_lon_wide[j]>15:
                n_large_wave+=1
        
        ### make sure the wave event is not in tropics ###
        n_north = 0
        for j in np.arange(len(track_lat)):
            if track_lat[j]>30:
                n_north+=1
                
        if day+1 >= Duration and n_large_wave>=Duration and n_north>=Duration:
            Blocking_lon.append(track_lon)
            Blocking_lat.append(track_lat)
            Blocking_date.append(track_date)
            Blocking_lon_wide.append(track_lon_wide)
            Blocking_area.append(track_area)
            Blocking_label.append(track_label)
            B_freq += B_count
            for dd in np.arange(len(B_count2)):
                B_freq2[d+dd,:,:]+= B_count2[dd]
        
        ### Once we successfully detected a wave event, we mark that with nan to avoid repeating ###
        for dd in np.arange(day+1):
            lon_d[d+dd][track_lon_index[dd]] = np.nan
            lat_d[d+dd][track_lat_index[dd]] = np.nan
    print(d)

print('embryo tracking done')

#%% Save results
import matplotlib.pyplot as plt
fig = plt.figure(figsize=[12,7])
plt.contourf(lon,lat_NH,B_freq, 20, extend="both", cmap='Reds') 
cb=plt.colorbar()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('embryo2days_Freq_daily_1979_2021.png')  
plt.show()
plt.close()

print('results saved!')

# S2 --------------------------------------------------------------
print('----------------------- part 2 -----------------------')

#%%
### get the blocking peaking date and location and wave activity ###
Embryo_date = []
Embryo_label = []

for n in np.arange(len(Blocking_date)):
    Embryo_date.append(Blocking_date[n])  ### the peaking date is the first date of the blocking event ###
    Embryo_label.append(Blocking_label[n]) ### the peaking label is the first label of the blocking event ###

#%% Save results
with open("/scratch/bell/hu1029/LGHW/Embryo2days_date", "wb") as fp:
    pickle.dump(Embryo_date, fp)
with open("/scratch/bell/hu1029/LGHW/Embryo2days_label", "wb") as fp:
    pickle.dump(Embryo_label, fp)


# %% filter the target region ------

# time management 
Datestamp = pd.date_range(start="1979-01-01", end="2021-12-31")
Date0 = pd.DataFrame({'date': pd.to_datetime(Datestamp)})
timestamp = list(Date0['date'])
timestamparr = np.array(timestamp)

typeid = 1
type_idx = typeid - 1
rgname = 'ATL'

# read in lat and lon for LWA
lat = np.load("/scratch/bell/hu1029/LGHW/LWA_lat_1979_2021_ERA5_6hr.npy")
lon = np.load("/scratch/bell/hu1029/LGHW/LWA_lon_1979_2021_ERA5_6hr.npy")
lat_mid = int(len(lat)/2) + 1 
Blklon = lon
if rgname == 'SP':
    Blklat = lat[lat_mid:len(lat)]
    print(Blklat)
else:
    Blklat = lat[0:lat_mid-1]
    print(Blklat)

# Check each blocking event, to see if it's within the target region
ATLlist = []
print(type_idx,flush=True)
print(f'Type{type_idx+1}, {rgname}, -------------filtering--------------',flush=True)
blocking_array = np.zeros((len(timestamp), len(Blklat), len(Blklon)),dtype=np.bool_)
blockingID_array = np.full((len(timestamp), len(Blklat), len(Blklon)),fill_value=-1, dtype=np.int32)

for event_idx in range(len(Embryo_date)):

    peakinglat = peakinglatList[event_idx]  # the peaking latitude
    peakinglon = peakinglonList[event_idx]  # the peaking longitude

    event_dates = Blocking_diversity_date[event_idx] # a list of dates

    timeindex = np.where(np.isin(timestamparr, event_dates))[0]  # find the time index in the total len
    blklabelarr = np.array(Blocking_diversity_label[type_idx][event_idx]) # the 3d array of blocking label

    # logic2: based on the peaking location
    lat_in = lat_min <= peakinglat <= lat_max
    if lon_min <= lon_max:
        lon_in = lon_min <= peakinglon <= lon_max
    else:
        lon_in = (peakinglon >= lon_min) or (peakinglon <= lon_max)
        
    if(lat_in and lon_in):
        ATLlist.append(event_idx)
        blocking_array[timeindex, :, :] = np.logical_or(blocking_array[timeindex, :, :],blklabelarr) # fill into the 3d array
        t_idx, y_idx, x_idx = np.where(blklabelarr > 0)
        abs_t_idx = timeindex[t_idx]
        blockingID_array[abs_t_idx, y_idx, x_idx] = int(event_idx) # fill into the event id values
        print('Event_idx in the target region: ',event_idx,flush=True)

    
# get the blocking days 
flagnum = np.nansum(blocking_array, axis=(1,2))

print(f"Total blocking length, Type{type_idx+1}_{rgname}: {len(Blockingday)}")
# save the blocking array 
np.save(f"/scratch/bell/hu1029/LGHW/embryo2days_FlagmaskClusters_Type{type_idx+1}_{rgname}.npy", blocking_array)
# save the blocking id list
with open(f"/scratch/bell/hu1029/LGHW/embryo2days_FlagmaskClustersEventList_Type{type_idx+1}_{rgname}", "wb") as fp:
    pickle.dump(ATLlist, fp)
# save the id array
np.save(f"/scratch/bell/hu1029/LGHW/BlockingClustersEventID_Type{type_idx+1}_{rgname}.npy", blockingID_array)

print('blockingarr saved ----------------',flush=True)