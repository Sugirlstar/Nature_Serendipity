def generate_colors(n, colormap='tab20'):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i / n) for i in range(n)]
    return colors

def create_Map(lon, lat, M, minv, maxv, interv=7, continterv = None, fill=True, ax=None, 
               leftlon=-180, rightlon=180, lowerlat=20, upperlat=90, centralLon=0, extend='both', alpha=1,
               fig=None, nrows=1, ncols=1, index=1, figsize=(15, 18),
               colr='coolwarm',contourcolr='gray', title=None, axisfontsize=12):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.path as mpath
    from matplotlib.lines import Line2D

    # lon = ((lon + 180) % 360) - 180
    # lon = lon-180
    proj = ccrs.PlateCarree(central_longitude=centralLon)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    # If not provided, create a new figure
    if fig is None:
        fig = plt.figure(figsize=figsize)   
    if ax == None:
        ax = fig.add_subplot(nrows, ncols, index, projection = proj)
        ax.set_extent(img_extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, 
                        color='gray',linestyle='--',alpha=0.5, fontsize=axisfontsize) #

    if fill==True:
        cf = ax.contourf(lon,lat,M,transform=ccrs.PlateCarree(),cmap=colr,levels=np.linspace(minv, maxv, interv), extend=extend, alpha=alpha)
    else:
        cf = ax.contour(lon,lat,M,transform=ccrs.PlateCarree(),colors=contourcolr,levels=np.arange(minv, maxv+continterv, continterv), negative_linestyles = 'dashed')
        ax.clabel(cf, inline=True, fontsize=8, fmt='%1.1f', colors=contourcolr, use_clabeltext=True)

    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    return fig, ax, cf


def calculate_grid_area_from_bounds(lat, lon, radius=6371):
    import numpy as np
    
    def extend_bounds(coords, min_val, max_val):
        dcoords = np.diff(coords) # calculate the difference between each point
        left = coords[0] - dcoords[0] / 2 # extend the left boundary
        right = coords[-1] + dcoords[-1] / 2 # extend the right boundary
        mid = coords[:-1] + dcoords / 2 # calculate the mid point
        # make the new set of coordinate points
        bounds = np.concatenate([[left], mid, [right]])
        cps = np.clip(bounds, min_val, max_val)
        return cps # don't need the orignial points

    lat = extend_bounds(lat, -90, 90)
    lon = extend_bounds(lon, 0, 360)
    lat_bounds = np.radians(lat)
    lon_bounds = np.radians(lon)

    # calculate the area of each grid
    area = np.zeros((len(lat)-1, len(lon)-1))
    for i in range(len(lat)-1):
        for j in range(len(lon)-1):
            lat_bottom = lat_bounds[i]
            lat_top = lat_bounds[i + 1]
            lon_left = lon_bounds[j]
            lon_right = lon_bounds[j + 1]
            area[i, j] = (
                radius**2
                * (lon_right - lon_left)
                * (np.sin(lat_top) - np.sin(lat_bottom))
            )
    area = area/1e6
    return area


def calculate_area(lat_min, lat_max, lon_min, lon_max, radius=6371):

    lat_min_rad = np.radians(lat_min)
    lat_max_rad = np.radians(lat_max)
    lat_fraction = np.abs(np.sin(lat_max_rad) - np.sin(lat_min_rad))
    lon_fraction = np.abs(lon_max - lon_min) / 360.0
    if lon_min > lon_max:
        lon_fraction = ((360-lon_min)+lon_max)/360.0
    area = 2 * np.pi * radius**2 * lat_fraction * lon_fraction
    return area
    
def addSegments(ax,points,colr='black',linewidth=2,labels=None,alpha=1,linestyle='solid'):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.path as mpath
    from matplotlib.lines import Line2D
    
    # segments = []
    # for i in range(len(points) - 1):
    #     segment = [points[i], points[i + 1]]
    #     segments.append(segment)

    segments = []
    for i in range(len(points) - 1):
        lon1, lat1 = points[i]
        lon2, lat2 = points[i + 1]
        if lon1<0 or lon2<0:
            print('Longitude should be within 0-360 degrees')
        # if the longitude crosses 0 degrees, we need to split the line into two segments
        if abs(lon1 - lon2) > 180:
            
            lonm1 = 0 if lon1 < abs(360 - lon1) else 360
            lonm2 = 0 if lon2 < abs(360 - lon2) else 360
            segments.append([(lon1, lat1), (lonm1, (lat1+lat2)/2)])
            segments.append([(lonm2, (lat1+lat2)/2), (lon2, lat2)])
        else:
            segments.append([(lon1, lat1), (lon2, lat2)])

    colr = [colr] * len(segments)
    
    for i, line_segment in enumerate(segments):
        
        lons = [p[0] for p in line_segment]
        lats = [p[1] for p in line_segment]        
        color = colr[i]  # get the color      
        line = Line2D(
            lons, lats, 
            color=color,  # set color
            linewidth=linewidth, 
            transform=ccrs.PlateCarree(),
            alpha=alpha,
            linestyle=linestyle
        )        
        ax.add_line(line)
        # if labels are provided, add them
        if labels is not None:
            # calculate the mid point of each segment
            mid_lon = (lons[0] + lons[1]) / 2
            mid_lat = (lats[0] + lats[1]) / 2
            # add the label
            ax.text(mid_lon, mid_lat, labels[i], fontsize=5, 
                ha='center', transform=ccrs.PlateCarree())

def addPatch(minlat,maxlat,minlon,maxlon,ax,colr='black',linewidth=2):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.path as mpath
    from matplotlib.lines import Line2D

    corners1 = [
        (minlon, minlat),  # left lower
        (maxlon, minlat),  # right lower
        (maxlon, maxlat),  # right upper
        (minlon, maxlat),  # left upper
    ]
    polygon1 = Polygon(
        corners1,
        closed=True,
        edgecolor=colr,
        facecolor='none',
        linewidth=linewidth,
        transform=ccrs.PlateCarree(),  
    )
    ax.add_patch(polygon1)

def create_Polarmap(lon, lat, M, minv, maxv, interv=7, continterv = None, fill=True, ax=None, 
               leftlon=0, rightlon=360, lowerlat=20, upperlat=90, centralLon=0, extend='both', alpha=1,
               fig=None, nrows=1, ncols=1, index=1, figsize=(15, 18),
               colr='coolwarm',contourcolr='gray', title=None):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import xarray as xr
    import numpy as np
    from matplotlib.patches import Polygon
    import matplotlib.path as mpath
    from matplotlib.lines import Line2D

    proj = ccrs.NorthPolarStereo(central_longitude=centralLon)
    img_extent = [leftlon, rightlon, lowerlat, upperlat]
    # If not provided, create a new figure
    if fig is None:
        fig = plt.figure(figsize=figsize)   
    if ax == None:
        ax = fig.add_subplot(nrows, ncols, index, projection = proj)
        ax.set_extent(img_extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('110m'))
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray',linestyle='--',alpha=0.5) #
        # make the boundary a circle
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.44
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if fill==True:
        cf = ax.contourf(lon,lat,M,transform=ccrs.PlateCarree(),cmap=colr,levels=np.linspace(minv, maxv, interv), extend=extend, alpha=alpha)
    else:
        cf = ax.contour(lon,lat,M,transform=ccrs.PlateCarree(),colors=contourcolr,levels=np.arange(minv, maxv, continterv), negative_linestyles = 'dashed')

    if title:
        ax.set_title(title)
    
    return fig, ax, cf

def readTRACKoutput_version152(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    track_data = []

    current_track_id = None
    track_points = []

    for line in lines:
        if line.startswith('TRACK_ID'):
            if current_track_id is not None:
                track_data.append((current_track_id, track_points)) # this is to save the last track
            current_track_id = int(line.split()[1])
            track_points = []
        
        elif line.strip() and len(line.split())==4:
            parts = line.split()
            timestep = int(parts[0])
            longitude = float(parts[1])
            latitude = float(parts[2])
            track_points.append((timestep, longitude, latitude))

    if current_track_id is not None: # this is to save the last track
        track_data.append((current_track_id, track_points))

    return track_data

def readTRACKoutput(file_path):

    with open(file_path, 'r') as f:
        lines = f.readlines()

    track_data = []

    current_track_id = None
    track_points = []

    for line in lines:
        if line.startswith('TRACK_ID'):
            if current_track_id is not None:
                track_data.append((current_track_id, track_points)) # this is to save the last track
            current_track_id = int(line.split()[1])
            track_points = []
        
        elif line.strip() and len(line.split())==8:
            parts = line.split()
            timestep = int(parts[0])
            longitude = float(parts[1])
            latitude = float(parts[2])
            track_points.append((timestep, longitude, latitude))

    if current_track_id is not None: # this is to save the last track
        track_data.append((current_track_id, track_points))

    return track_data

def distributionPlot(Plot,lon,lat,ptname):
    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter 

    lon=lon-180

    levs = 15
    fig = plt.figure(figsize=[12,7])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    plt.contourf(lon[:],lat[:],Plot, levs, transform=ccrs.PlateCarree(central_longitude=180), extend="both", cmap='RdBu_r')  
    ax.coastlines(linestyle="--", alpha=0.3)
    ax.gridlines(linestyle="--", alpha=0.1)

    lon_formatter = LongitudeFormatter(transform_precision = 1,zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    intgap1 = round((np.nanmax(lon)-np.nanmin(lon))/5,0)
    intgap2 = round((np.nanmax(lat)-np.nanmin(lat))/5,0)
    xtic = np.arange(round(np.nanmin(lon),0),round(np.nanmax(lon),0),intgap1)
    ytic = np.arange(round(np.nanmin(lat),0),round(np.nanmax(lat),0),intgap2)
    ax.set_xticks(xtic, crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_yticks(ytic, crs=ccrs.PlateCarree(central_longitude=180))

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cb=plt.colorbar()
    plt.savefig(ptname+'.png')
    plt.close() 

def distributionPlotwithCenter(Plot,lon,lat,lon_star,lat_star,ptname):
    import matplotlib.pyplot as plt
    import numpy as np
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter 

    lon=lon-180

    levs = 15
    fig = plt.figure(figsize=[12,7])
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    plt.contourf(lon[:],lat[:],Plot, levs, transform=ccrs.PlateCarree(central_longitude=180), extend="both", cmap='RdBu_r')  
    ax.coastlines(linestyle="--", alpha=0.3)
    ax.gridlines(linestyle="--", alpha=0.1)

    lon_formatter = LongitudeFormatter(transform_precision = 1,zero_direction_label=False)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    intgap1 = round((np.nanmax(lon)-np.nanmin(lon))/5,0)
    intgap2 = round((np.nanmax(lat)-np.nanmin(lat))/5,0)
    xtic = np.arange(round(np.nanmin(lon),0),round(np.nanmax(lon),0),intgap1)
    ytic = np.arange(round(np.nanmin(lat),0),round(np.nanmax(lat),0),intgap2)
    ax.set_xticks(xtic, crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_yticks(ytic, crs=ccrs.PlateCarree(central_longitude=180))
    ax.plot(lon_star, lat_star, marker='*', color='red', markersize=6, transform=ccrs.PlateCarree())

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    cb=plt.colorbar()
    plt.savefig(ptname+'.png')
    plt.close() 

def calculate_composites2(pos_HW, matriz):
    import numpy as np
    from scipy import stats
    dic_composites = {} # dic_composites to store the indices of each time lag

    time_lags = np.arange(-20, 21, 1) # time_lags define the range of time lags from -20 to 20
    for pos in pos_HW.index:
        if pos == 0: 
            dic_composites[0] = [] # if pos is 0, initialize dic_composites[0]
            dic_composites[0].append(int(pos_HW.iloc[0][0])) # and add the index of the first heatwave event
            continue
        #elif pos_HW.iloc[pos][0] >= 27270: break  # len(other variables)
        elif pos_HW.iloc[pos][0] == matriz.shape[0]-20: break
        elif pos == pos_HW.index[-1]:
            break # check if it exceeds the matrix range or is the last index, if so, exit the loop
        elif pos_HW.iloc[pos][0]-1 != pos_HW.iloc[pos-1][0]: # if the current index and the previous index are not continuous, it means a new HW starts
            for num in time_lags:  
                if num in dic_composites.keys(): dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))  
                else: 
                    dic_composites[num] = []
                    dic_composites[num].append(int(pos_HW.iloc[pos][0]+num))

    composites_matrix_complete =  np.zeros((len(time_lags), matriz.shape[1], matriz.shape[2]))
    for ii, lag in enumerate(np.sort(list(dic_composites))):
        composites_matrix_complete[ii] = np.mean(matriz[dic_composites[lag]], axis = 0)

    # day_20 = np.mean(matriz[dic_composites[-10]], axis=0)
    # day_15 = np.mean(matriz[dic_composites[-5]], axis=0)
    # day_10 = np.mean(matriz[dic_composites[0]], axis=0)
    # day_5 = np.mean(matriz[dic_composites[5]], axis=0)
    # day_0 = np.mean(matriz[dic_composites[10]], axis=0)
    # day_5_post = np.mean(matriz[dic_composites[15]], axis=0)

    day_0 = np.mean(matriz[dic_composites[0]], axis=0)
    day_5 = np.mean(matriz[dic_composites[-5]], axis=0)
    day_10 = np.mean(matriz[dic_composites[-10]], axis=0)
    day_15 = np.mean(matriz[dic_composites[-15]], axis=0)
    day_20 = np.mean(matriz[dic_composites[-20]], axis=0)
    day_5_post = np.mean(matriz[dic_composites[5]], axis=0)

    composites_matrix = np.zeros((6, day_0.shape[0], day_0.shape[1]))
    for i, matriz_i in enumerate([day_20, day_15, day_10, day_5, day_0, day_5_post]):
        composites_matrix[i,:,:] = matriz_i
    
    return composites_matrix, composites_matrix_complete

def spatial_smoothing_wrap(array, x):
    import numpy as np
    from scipy.ndimage import convolve
    # Create a mask to identify the locations of np.nan
    nan_mask = np.isnan(array)
    # Replace np.nan with 0
    array_filled = np.where(nan_mask, 0, array)
    # Extend the columns of the array so that the first and last columns are connected
    extended_array = np.hstack((array_filled[:, -(x//2):], array_filled, array_filled[:, :x//2]))
    extended_mask = np.hstack((~nan_mask[:, -(x//2):], ~nan_mask, ~nan_mask[:, :x//2]))
    # Create a x*x average smoothing kernel
    kernel = np.ones((x, x))
    # Perform convolution operations on the extended array and mask
    smoothed_extended_array = convolve(extended_array, kernel, mode='reflect')
    smoothed_extended_mask = convolve(extended_mask.astype(float), kernel, mode='reflect')
    # Avoid division by 0, set a minimum value
    smoothed_extended_mask = np.where(smoothed_extended_mask == 0, np.nan, smoothed_extended_mask)
    # Calculate the final smoothing result
    smoothed_array = smoothed_extended_array / smoothed_extended_mask
    # Crop back to the original size
    smoothed_array = smoothed_array[:, x//2:-(x//2)]
    # Restore the original np.nan
    smoothed_array[nan_mask] = np.nan
    return smoothed_array

def anomalies_seasons(df_VAR):
    import numpy as np
    import pandas as pd
    import datetime as dt
    ANOMA = df_VAR * np.nan

    def custom_strptime(time_data):
        if len(time_data) == 7:
            month = time_data[:2]
            day = time_data[2:4]
            year = '0' + time_data[4:]
            date_str = f'{month}{day}{year}'
            return dt.datetime.strptime(date_str, '%m%d%Y')
        else:
            return dt.datetime.strptime(time_data, '%m%d%Y')
    
    # dates_d = np.array([dt.datetime.strptime(iii, '%m%d%Y') for iii in df_VAR.index])
    dates_d = np.array([custom_strptime(iii) for iii in df_VAR.index])
    # Iterate over each month from January (1) to December (12):
    for i in np.arange(1, 13):
        mes = df_VAR.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] # get all data for the i-th month from df_VAR and store it in mes
        dates_d_mes = dates_d[np.where(np.array([ii.month for ii in dates_d])==i)[0]] # get all dates for the i-th month from dates_d and store it in dates_d_mes
        temp = mes * np.nan
        Nodays = np.array([ii.day for ii in dates_d_mes]).max()
        if np.isnan(Nodays) == True: continue
        # Loop through each day of the current month, from the 1st to the maximum number of days
        for j in np.arange(1, Nodays + 1):
            # Select the data from mes that belongs to the current day and store it in dia, which is all years of the i-th month and j-th day
            dia = mes.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]]
            media = dia.mean() # get the mean for the i-th month and j-th day
            anoma = dia - media
            temp.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]] = anoma
        ANOMA.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] = temp
    return ANOMA

def center_white_anom(cmap, num, bounds, limite):
    import matplotlib as mpl
    import numpy as np
    import matplotlib.cm 
    barcmap = matplotlib.cm.get_cmap(cmap, num)
    barcmap.set_bad(color='white', alpha=0.5)
    bar_vals = barcmap(np.arange(num))  # extract those values as an array
    pos = np.arange(num)
    centro = pos[(bounds >= -limite) & (bounds <= limite)]
    for i in centro:
        bar_vals[i] = [1, 1, 1, 1]  # change the middle value
    newcmap = mpl.colors.LinearSegmentedColormap.from_list("new" + cmap, bar_vals)
    return newcmap

def format_latitude(value, pos):
    # Format latitude ticks with degree symbol and 'N'
    if value >= 0:
        return f'{value:.0f}°N'
    else:
        return f'{-value:.0f}°S'

def clickGetGrid(lat_min,lat_max,lon_min,lon_max,lat_res,lon_res,A,noA=False):

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import random
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # create lat lon grids 
    lons = np.arange(lon_min, lon_max, lon_res)
    lats = np.arange(lat_min, lat_max, lat_res)
    # lat_grid, lon_grid = np.meshgrid( lats, lons)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    if(noA==False):
        data = np.random.rand(len(lats), len(lons)) # generate a 2D array filled with random values
    else:
        data = A

    # map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS)

    # plot the data
    cs = ax.pcolormesh(lon_grid, lat_grid, data, cmap='viridis', transform=ccrs.PlateCarree(central_longitude=180))
    plt.colorbar(cs, ax=ax, orientation='vertical')
    #img_extent = [lon_min, lon_max, lat_min, lat_max]
    #cs = ax.imshow(data, extent=img_extent, origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')
    #plt.colorbar(cs, ax=ax, orientation='vertical')

    # click event handler
    def onclick(event):
        if event.inaxes == ax:
            lon, lat = event.xdata, event.ydata
            n = int((lat - lat_min) / lat_res)
            m = int((lon - lon_min) / lon_res)
            
            print(f"Clicked at: lon={lon:.2f}, lat={lat:.2f}")
            print(f"Array indices: n={n}, m={m}")
            print(f"Array value: {data[n, m]}")
            print("---------------")
            
            # plot star at clicked location
            ax.plot(lon, lat, marker='*', color='red', markersize=10, transform=ccrs.PlateCarree(central_longitude=180))
            plt.draw()

    # click connection
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

def CLIMA(df_VAR):
    import numpy as np
    import pandas as pd
    import datetime as dt
    CLIMA = df_VAR * np.nan

    def custom_strptime(time_data):
        if len(time_data) == 7:
            month = time_data[:2]
            day = time_data[2:4]
            year = '0' + time_data[4:]
            date_str = f'{month}{day}{year}'
            return dt.datetime.strptime(date_str, '%m%d%Y')
        else:
            return dt.datetime.strptime(time_data, '%m%d%Y')
    
    # dates_d = np.array([dt.datetime.strptime(iii, '%m%d%Y') for iii in df_VAR.index])
    dates_d = np.array([custom_strptime(iii) for iii in df_VAR.index])

    for i in np.arange(1, 13):
        mes = df_VAR.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]]
        temp = mes * np.nan
        dates_d_mes = dates_d[np.where(np.array([ii.month for ii in dates_d])==i)[0]]
        Nodays = np.array([ii.day for ii in dates_d_mes]).max()
        if np.isnan(Nodays) == True: continue
        for j in np.arange(1, Nodays + 1):
            dia = mes.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]]
            media = pd.DataFrame(np.tile(dia.mean(), (dia.shape[0],1)), index = dia.index)
            temp.iloc[np.where(np.array([ii.day for ii in dates_d_mes])==j)[0]] = media
        CLIMA.iloc[np.where(np.array([ii.month for ii in dates_d])==i)[0]] = temp
    return CLIMA

def subseasonal_anomalies(df_VAR):
    import numpy as np
    import pandas as pd
    import datetime as dt

    def custom_strptime(time_data):
        if len(time_data) == 7:
            month = time_data[:2]
            day = time_data[2:4]
            year = '0' + time_data[4:]
            date_str = f'{month}{day}{year}'
            return dt.datetime.strptime(date_str, '%m%d%Y')
        else:
            return dt.datetime.strptime(time_data, '%m%d%Y')
    
    # dates_d = np.array([dt.datetime.strptime(iii, '%m%d%Y') for iii in df_VAR.index])
    dates_d = np.array([custom_strptime(iii) for iii in df_VAR.index])

    years = np.unique(np.array([ii.year for ii in dates_d]))
    ANOMA = df_VAR * np.nan
    for i in years:
        pos_y = np.where(np.array([ii.year for ii in dates_d])==i)[0]
        year = df_VAR.iloc[pos_y] # get all data for the i-th year from df_VAR
        dates_d_year = dates_d[np.where(np.array([ii.year for ii in dates_d])==i)[0]] # get all dates for the i-th year from dates_d
        temp = year * np.nan
        for j in ([3,4,5], [6,7,8], [9,10,11]):
            pos_season = np.where((np.array([ii.month for ii in dates_d_year])==j[0]) | (np.array([ii.month for ii in dates_d_year])==j[1]) | (np.array([ii.month for ii in dates_d_year])==j[2]))[0]
            season = year.iloc[pos_season] # get the indices pos_season and data season for the current season j (three months) in the current year
            media = season.mean()
            anoma = season - media
            temp.iloc[pos_season] = anoma
        ANOMA.iloc[pos_y] = temp
    return ANOMA