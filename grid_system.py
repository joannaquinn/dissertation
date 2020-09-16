import os
import datetime as dt
import csv
from chardet import detect
import random
import math
import numpy as np
import pandas as pd

from basic_imports import *
from misc_utils import *
from constants_jo import *
from load_files import *
from download_data import *
from preprocessing import time_from_timestamp

from model_names import INTERVALS, CELL_SIZES, GRID_DIMS

def dict_from_txt(filename):
    '''
    dictionary of road/landuse from text file
    formatas follows
    0:
    '''
    out = {}
    with open(filename, 'r') as f:
        for line in f:
            idd = line.split(":")[0]
            values = line.split(":")[1]
            for value in values.split(","):
                if value != "":
                    out[value.strip()] = int(idd.strip())
    return out

# constants
ROADS = dict_from_txt('roads.txt')
LANDUSE = dict_from_txt('landuse.txt')
TRAFFIC = {'traffic_signals': 1}

def cut_df(df, start, end, timestamp_col = 'timestamp'):
    '''
    given a start and end time as timestamps, 
    returns sliced dataframe
    '''
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    new_df = df[df[timestamp_col] < str(end)]
    new_df = new_df[new_df[timestamp_col] >= str(start)]
    return new_df

def cut_df_time(df, start_time, end_time, timestamp_col = 'timestamp'):
    '''
    given a start and end time in format ('15:00:00', string)
    returns subset dataframe
    '''
    df2 = df.copy()
    df2['time'] = df2.apply(lambda x: time_from_timestamp(x[timestamp_col]), axis=1)
    start_time = pd.Timestamp(f'2001-01-01 {start_time}')
    end_time = pd.Timestamp(f'2001-01-01 {end_time}')
    return cut_df(df2, start_time, end_time, timestamp_col='time')

def mean_value_for_cell(df, grid_cell_id, col='pm2_5'):
    '''
    given a dataframe (with cell IDs specified) and a specific cell ID,
    returns a mean value for that cell (used for PM levels)
    '''
    new_df = df[df['id'] == grid_cell_id]
    out = new_df[col].mean()
    return out

def max_value_for_cell(df, grid_cell_id, col='fclass', nan_value=0):
    '''
    given a dataframe (with cell IDs specified) and a specific cell ID,
    returns a max value for that cell (used for road type)
    '''
    new_df = df[df['id'] == grid_cell_id]
    out = new_df[col].max()
    if out is np.nan:
        out = nan_value
    return out

def max_area_value_for_cell(df, grid_cell_id, col='fclass', area_col='area'):
    '''
    given a dataframe (with cell IDs specified) and a specific cell ID,
    returns the value from col which corresponds to max area
    '''
    new_df = df[df['id'] == grid_cell_id]
    if len(new_df[area_col]) != 0:
        idx = new_df[area_col].idxmax()
        fclass = df[col].iloc[idx]
        return fclass
    else:
        return None

def pm_grid_from_df(df, grid_dim, col='pm2_5', interval=15, nans=False, 
                 start_time=None, end_time=None, filename=None, indices_filename = None,
                 progress=True, order='F', timestamp_col = 'timestamp'):
    '''
    creates an array of dim T x rows x cols where
    T is number of timesteps (set using the interval in minutes and start/end times),
    rows/cols is the number of rows/cols in grid, ie grid_dim[0] and grid_dim[1] repectively 
    missing values will be zero, otherwise nan
    grid cell id starts indexing at 1 and is contained in 'id' column of dataframe
    save to .npy file
    if copy_missing_intervals, set missing timesteps to prev one (only use if there is little missing data)
    '''
    # get start and end timestamps
    if start_time is None:
        start_time = df[timestamp_col].min()
    if end_time is None:
        end_time = df[timestamp_col].max()
    
    # # data will be in intervals of {interval} minutes; 
    # timestamp will indicate start time of each interval
    start_time_rounded = pd.Timestamp(start_time).floor(freq = f'{interval}Min')
    end_time_rounded = pd.Timestamp(end_time).ceil(freq = f'{interval}Min')
    
    time_dim = int(abs((start_time_rounded-end_time_rounded).total_seconds() / (60*interval)))
    time_indices = [start_time_rounded + (i*dt.timedelta(minutes=interval)) for i in range(time_dim)]
    
    print(f'Data runs from {start_time_rounded} to {end_time_rounded}')
    
    # initialise array
    pm_array = np.zeros((time_dim, grid_dim[0], grid_dim[1]))
    print(f'Array will be of size {pm_array.shape}')
    grid_cell_ids = list(range(1, grid_dim[0]*grid_dim[1]+1))
    
    i = -1 # timestamp counter
    for t in time_indices:
        i += 1
        #c check progress
        if progress:
            if i % 50 == 0:
                print(f'Timestamp {i} out of {time_dim} {int(i*100/time_dim)} percent complete')
        
        # get sliced dataframe
        cut_dff = cut_df(df, t, t + dt.timedelta(minutes=interval), timestamp_col=timestamp_col)
        
        # loop through each grid cell
        for grid_cell_id in grid_cell_ids:
            np_index = np.unravel_index(grid_cell_id-1, shape=grid_dim, order=order)
            pm_level = mean_value_for_cell(cut_dff, grid_cell_id, col=col)
            
            #set pm_level to zero if there is no data available
            if nans == False:
                if np.isnan(pm_level):
                    pm_level = 0
            
            pm_array[i, np_index[0], np_index[1]] = pm_level
    
        if np.count_nonzero(pm_array[i]) == 0:
            print(f'Timestamp {t} (index {i}) has no PM data')
        # else:
        #     print(f'Timestamp {t} (index {i}) has {np.count_nonzero(pm_array[i])} data')
    
    # save array to file
    if filename is not None:
        np.save(filename, pm_array)
        
    if indices_filename is not None:
        np.savetxt(indices_filename, time_indices, fmt='%s')
    
    print("Finished!")
    return pm_array, time_indices


def pm_grid_from_file(filename, indices_filename=None):
    if indices_filename is None:
        return np.load(filename)
    else:
        return np.load(filename), list(pd.read_csv(indices_filename, header=None,parse_dates=[0]).values.flatten())
    
# def grid_df_from_grid(grid):
#     '''
#     converts (temporal) grid as defined above into pandas df with columns
#     t (timestep), i (first np coordinate), j (second np coordinate)
#     value = value in grid (usually PM)
#     '''
#     grid_dim = grid.shape
#     timesteps = np.array(range(grid_dim[0]))
    
#     dfs = []
#     for t in timesteps:
#         i_coords, j_coords = grid[t].nonzero()
#         values = grid[t][(i_coords, j_coords)]
#         df = pd.DataFrame(data={'t':t,
#                                 'i':i_coords,
#                                 'j':j_coords,
#                                 'value':values})
#         dfs.append(df)
#     return pd.concat(dfs)

# def grid_from_grid_df(df, nans=False):
#     '''
#     converts grid style df to np array
#     '''
#     timesteps = df['t'].unique().sort_values(ascending=True)
#     grid_dim = (timesteps[-1], int(df[i].max()), int(df[j].max()))
    
#     grid = np.zeros(grid_dim)
#     for t in timesteps:
#         g = df[df['t']== t].pivot('i', 'j', 'value').values
#         if nans == False:
#             g[np.isnan(g)] = 0 # remove nans
#         grid[t] = g.copy()
    
#     return grid

def rel_humidity_grid_from_df(df, grid_dim, humid_col='humidity', temp_col='temperature', interval=15, nans=False, 
                 start_time=None, end_time=None, filename=None, indices_filename = None,
                 progress=False, order='F', timestamp_col = 'timestamp'):
    '''
    creates an array of dim T x rows x cols where
    T is number of timesteps (set using the interval in minutes and start/end times),
    rows/cols is the number of rows/cols in grid, ie grid_dim[0] and grid_dim[1] repectively 
    will obtain avg temp and avg humidity, and set every grid cell for that timestamp equal to ratio humid/temp
    grid cell id starts indexing at 1 and is contained in 'id' column of dataframe
    save to .npy file
    '''
    # get start and end timestamps
    if start_time is None:
        start_time = df[timestamp_col].min()
    if end_time is None:
        end_time = df[timestamp_col].max()
    
    # # data will be in intervals of {interval} minutes; 
    # timestamp will indicate start time of each interval
    start_time_rounded = pd.Timestamp(start_time).floor(freq = f'{interval}Min')
    end_time_rounded = pd.Timestamp(end_time).ceil(freq = f'{interval}Min')
    
    time_dim = int(abs((start_time_rounded-end_time_rounded).total_seconds() / (60*interval)))
    time_indices = [start_time_rounded + (i*dt.timedelta(minutes=interval)) for i in range(time_dim)]
    
    print(f'Data runs from {start_time_rounded} to {end_time_rounded}')
    
    # initialise array
    humidity_array = np.zeros((time_dim, grid_dim[0], grid_dim[1]))
    print(f'Array will be of size {humidity_array.shape}')
    
    i = -1 # timestamp counter
    for t in time_indices:
        i += 1
        #c check progress
        if progress:
            if i % 50 == 0:
                print(f'Timestamp {i} out of {time_dim} {int(i*100/time_dim)} percent complete')
        
        # get sliced dataframe
        cut_dff = cut_df(df, t, t + dt.timedelta(minutes=interval), timestamp_col=timestamp_col)
        
        # get avg relative humidity
        temp = cut_dff[temp_col].mean()
        humidity = cut_dff[humid_col].mean()
        assert temp != 0, 'average temperature is 0! terminating'
        assert humidity != 0, 'average humidity is 0! terminating'
        rel_humidity = humidity/temp
        
        humidity_array[i] = np.ones((humidity_array.shape[1], humidity_array.shape[2]))*rel_humidity
    
    # save array to file
    if filename is not None:
        np.save(filename, humidity_array)
        
    if indices_filename is not None:
        np.savetxt(indices_filename, time_indices, fmt='%s')
        
    return humidity_array, time_indices

def roads_grid_from_df(df, grid_dim, col='fclass', nans=False, filename=None, order='F'):
    '''
    creates an array of dim rows x cols where
    rows/cols is the number of rows/cols in grid, ie grid_dim[0] and grid_dim[1] repectively 
    will obtain max road type for each cell. if no info, will use minimum road type (0)
    grid cell id starts indexing at 1 and is contained in 'id' column of dataframe
    save to .npy file
    '''
    # initialise array
    road_array = np.zeros((grid_dim[0], grid_dim[1]))
    print(f'Array will be of size {road_array.shape}')
    grid_cell_ids = list(range(1, grid_dim[0]*grid_dim[1]+1))
    
    # map fclass to road ids
    df['roadid']= df[col].map(ROADS)
    
    for grid_cell_id in grid_cell_ids:
        np_index = np.unravel_index(grid_cell_id-1, shape=grid_dim, order=order)
        fclass = max_value_for_cell(df, grid_cell_id, col='roadid', nan_value=0)
        road_array[np_index] = fclass
    
    # save array to file
    if filename is not None:
        np.save(filename, road_array)
        
    return road_array

def traffic_grid_from_df(df, grid_dim, col='fclass', nans=False, filename=None, order='F'):
    '''
    creates an array of dim rows x cols where
    rows/cols is the number of rows/cols in grid, ie grid_dim[0] and grid_dim[1] repectively 
    will obtain 1 or 0 depending on whether traffic signal junction in cell
    grid cell id starts indexing at 1 and is contained in 'id' column of dataframe
    save to .npy file
    '''
    # initialise array
    traffic_array = np.zeros((grid_dim[0], grid_dim[1]))
    print(f'Array will be of size {traffic_array.shape}')
    grid_cell_ids = list(range(1, grid_dim[0]*grid_dim[1]+1))
    
    # map fclass to traffic ids
    df['trafficid']= df[col].map(TRAFFIC)
    df['trafficid'] = df['trafficid'].fillna(0)
    
    for grid_cell_id in grid_cell_ids:
        np_index = np.unravel_index(grid_cell_id-1, shape=grid_dim, order=order)
        fclass = max_value_for_cell(df, grid_cell_id, col='trafficid', nan_value=0)
        traffic_array[np_index] = fclass
    
    # save array to file
    if filename is not None:
        np.save(filename, traffic_array)
        
    return traffic_array

def landuse_grid_from_df(df, grid_dim, col='fclass', nans=False, 
                         green_filename=None, resi_filename=None, comm_filename=None, order='F'):
    '''
    creates 3 one hot arrays of dim rows x cols where
    rows/cols is the number of rows/cols in grid, ie grid_dim[0] and grid_dim[1] repectively 
    will obtain 1 or 0 depending on whether certain landuse typess are in cell (see landuse.txt for more info)
    green space, residential and commercial
    grid cell id starts indexing at 1 and is contained in 'id' column of dataframe
    save to .npy files
    '''
    # initialise arrays
    landuse_arrays = [np.zeros((grid_dim[0], grid_dim[1])) for _ in range(3)]
    print(f'Arrays will be of size {[landuse_arrays[j].shape for j in range(3)]}')
    grid_cell_ids = list(range(1, grid_dim[0]*grid_dim[1]+1))
    
    # map fclass to traffic ids
    df['landuseid']= df[col].map(LANDUSE)
    
    for grid_cell_id in grid_cell_ids:
        np_index = np.unravel_index(grid_cell_id-1, shape=grid_dim, order=order)
        landuseid = max_area_value_for_cell(df, grid_cell_id, col='landuseid')
        if (landuseid is not None) and not(np.isnan(landuseid)):
            landuse_arrays[int(landuseid)][np_index] = 1
    
    # save arrays to file
    j = 0
    for filename in (green_filename, resi_filename, comm_filename):
        if filename is not None:
            np.save(filename, landuse_arrays[j])
        j += 1
        
    return landuse_arrays

def plot_grid_time_hist(grid, indices, plot_filename=None, figsize=(15,10), tick_spacing=10):
    '''
    plots histogram of number of datapoints in each time interval
    for a PM grid of size (T, H, W)
    grid and indices loaded using pm_grid_from_file
    '''
    x = [pd.to_datetime(i) for i in indices]
    y = []
    for t in range(grid.shape[0]): # count number of non zero values for each timestamp
        val = len(grid[t].nonzero()[0])
        y.append(val)
        if val == 0:
            print(f'Missing data at {t}th index ({indices[t]})')
            
    assert len(indices) == len(y), 'indices must match grid'
        
    sns.set()
    fig, ax = plt.subplots(figsize=figsize)
    plot = sns.barplot(x=x, y=y, ax=ax)
    ax.xaxis_date()
    
    for tick in ax.get_xticklabels(): # adjust tick labels for readability
        tick.set_rotation(65);
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % tick_spacing != 0:
            label.set_visible(False)
            
    ax.set_xlabel('Time interval')
    ax.set_ylabel('Number of cells with data')
    
    if plot_filename is not None:
        fig = plot.get_figure()
        fig.savefig(plot_filename)
        
def plot_grid_time_hist_all(project_name, suffix='', tick_spacing=10, figsize=(15,10)):
    '''
    wrapper for plot grid time hist
    '''
    data_dir = project_mapping[project_name][2]
    plot_dir = project_mapping[project_name][3]
    
    labels = ('static','personal')
    intervals = INTERVALS['cnn'][project_name]
    cell_sizes = CELL_SIZES[project_name]
    
    for label in labels:
        for interval in intervals:
            for cell_size in cell_sizes:
                filename = os.path.join(data_dir, 'grid', f'{project_name}_{label}_{cell_size}m_{interval}min_pm2_5.npy')
                indices_filename = os.path.join(data_dir, 'grid', f'{project_name}_{label}_{interval}min.txt')
                grid, indices = pm_grid_from_file(filename, indices_filename=indices_filename)
                
                plot_filename = os.path.join(plot_dir, 'grid', f'{project_name}_{label}_{cell_size}m_{interval}min_interval_hist{suffix}.png')
                plot_grid_time_hist(grid, indices, plot_filename=plot_filename, figsize=figsize, tick_spacing=tick_spacing)
                
def create_pm_grids_wrapper(static_df, personal_df, location, interval, cell_size, timestamp_col='time', pms=('pm1','pm2_5','pm10'), has_humidity=True):
    grid_dim = GRID_DIMS[location][cell_size]
    data_dir = project_mapping[location][2]
    for pm in pms:
        filename = os.path.join(data_dir, 'grid', f'{location}_static_{cell_size}m_{interval}min_{pm}.npy')
        indices_filename = os.path.join(data_dir, 'grid', f'{location}_static_{interval}min.txt')
        static_pm_grid = pm_grid_from_df(static_df, grid_dim, col=pm, interval=interval, timestamp_col=timestamp_col,
                                         filename=filename, indices_filename=indices_filename)
        
        filename = os.path.join(data_dir, 'grid', f'{location}_personal_{cell_size}m_{interval}min_{pm}.npy')
        indices_filename = os.path.join(data_dir, 'grid', f'{location}_personal_{interval}min.txt')
        static_pm_grid = pm_grid_from_df(personal_df, grid_dim, col=pm, interval=interval, timestamp_col=timestamp_col,
                                        filename=filename, indices_filename=indices_filename)
        
    filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_humidity.npy')
    if has_humidity:
        humidity_grid, _ = rel_humidity_grid_from_df(static_df, grid_dim, interval=interval, filename=filename, timestamp_col=timestamp_col)
    else:
        humidity_grid = np.zeros(static_pm_grid.shape)
        np.save(filename, humidity_grid)
        
    

def create_nnet_dataset(location, interval, cell_size, net='cnn', has_humidity=True, pms=('pm2_5',), copy_missing_intervals=False):
    '''
    automatically creates nn inputs from gis .csv files
    get saved in corresponding folder (to manually sort into training/valid/test)
    '''
    # get data
    gis_dir = project_mapping[location][4]
    data_dir = project_mapping[location][2]
    roads_df = pd.read_csv(os.path.join(gis_dir, f'{location}_roads_{cell_size}.csv'))
    traffic_df = pd.read_csv(os.path.join(gis_dir, f'{location}_traffic_{cell_size}.csv'))
    landuse_df = pd.read_csv(os.path.join(gis_dir, f'{location}_landuse_{cell_size}.csv'))
    
    grid_dim = GRID_DIMS[location][cell_size]
    print(f'grid dim is {grid_dim}')
    
    # pm
    pm_grids = {'static':{}, 'personal':{}}
    for pm in pms:
        for label in ('static','personal'):
            filename = os.path.join(data_dir, 'grid', f'{location}_{label}_{cell_size}m_{interval}min_{pm}.npy')
            pm_grids[label][pm] = np.load(filename)
        assert pm_grids['static'][pm].shape == pm_grids['personal'][pm].shape, 'Static and PM grids not same shape'
        T = pm_grids['static'][pm].shape[0]
    
    # humidity: get average humidity over timestamp and set to constant across whole grid
    # humid and temp only recorded on static sensors
    if has_humidity:
        filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_humidity.npy')
        humidity_grid = np.load(filename)
    else:
        humidity_grid = np.zeros((T, grid_dim[0], grid_dim[1]))
        
    # roads
    filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_roads.npy')
    roads_grid = roads_grid_from_df(roads_df, grid_dim, col='fclass', filename=filename)
    
    # traffic
    filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_traffic.npy')
    traffic_grid = traffic_grid_from_df(traffic_df, grid_dim, col='fclass', filename=filename)
    
    # landuse
    green_filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_green.npy')
    resi_filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_resi.npy')
    comm_filename = os.path.join(data_dir, 'grid', f'{location}_{cell_size}m_{interval}min_comm.npy')
    landuse_grids = landuse_grid_from_df(landuse_df, grid_dim, col='fclass', green_filename=green_filename,resi_filename=resi_filename, comm_filename=comm_filename)
            
    # combine inputs
    output_dir = os.path.join(data_dir, f'{location}_{net}_{interval}min_{cell_size}')
    num_pm = len(pms)
    output_grids = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for t in range(T):
        filename = os.path.join(output_dir, f'{location}_{cell_size}m_{interval}min_{t}.npy')
        static_grids = [pm_grids['static'][pm][t] for pm in pms]
        personal_grids = [pm_grids['personal'][pm][t] for pm in pms]
        if (np.count_nonzero(personal_grids[0]) == 0) or (np.count_nonzero(static_grids[0]) == 0):
            print(f'Missing data at timestep {t}')
            if copy_missing_intervals:
                print(f'Copying data from timestep {t-1}')
                output_grid = output_grids[-1]
        else:
            output_grid = np.stack([static_grids[i] for i in range(num_pm)] + 
                                   [personal_grids[i] for i in range(num_pm)] + 
                                   [humidity_grid[t], roads_grid] + 
                                   [landuse_grid for landuse_grid in landuse_grids] + 
                                   [traffic_grid,])
        output_grids.append(output_grid)
        np.save(filename, output_grid)
        print(f'Saving {filename}')